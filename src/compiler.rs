use std::collections::{HashMap, HashSet};
use std::env;
use crate::{
    asm::{
        instrs_to_string, Arg32, Arg64, BinArgs, CMov, Instr, Loc, MemRef, MovArgs, Offset,
        Reg::{self, *},
        Reg32
    },
    mref,
    syntax::{Expr, FunDecl, Op1, Op2, Prog, Symbol},
};

struct Session {
    tag: u32,
    instrs: Vec<Instr>,
    funs: HashMap<Symbol, usize>,
}

const INVALID_ARG: &str = "invalid_argument";
const OVERFLOW: &str = "overflow";

const STACK_BASE: Reg = Rbx;
const INPUT_REG: Reg = R13;

#[derive(Debug, Clone)]
struct Ctxt<'a> {
    env: im::HashMap<Symbol, MemRef>,
    si: u32,
    curr_lbl: Option<&'a str>,
    in_fun: bool,
    tail_pos: bool,
    in_tail_loop: bool,
}

impl<'a> Ctxt<'a> {
    fn new() -> Ctxt<'a> {
        Ctxt {
            si: 0,
            curr_lbl: None,
            env: im::HashMap::default(),
            in_fun: false,
            tail_pos: false,
            in_tail_loop: false,
        }
    }

    fn with_params(params: &[Symbol]) -> Ctxt<'a> {
        let env = params
            .iter()
            .enumerate()
            .map(|(i, param)| (*param, mref![Rbp + %(8 * (i + 2))]))
            .collect();
        Ctxt {
            si: 0,
            curr_lbl: None,
            env,
            in_fun: true,
            tail_pos: false,
            in_tail_loop: false,
        }
    }

    fn with_tail_pos(&self, tail_pos: bool) -> Ctxt<'a> {
        Ctxt {
            tail_pos: tail_pos,
            ..self.clone()
        }
    }

    fn with_tail_pos_and_loop(&self, tail_pos: bool, in_tail_loop: bool) -> Ctxt<'a> {
        Ctxt {
            tail_pos: tail_pos,
            in_tail_loop: in_tail_loop,
            ..self.clone()
        }
    }

    fn lookup(&self, x: Symbol) -> MemRef {
        *self
            .env
            .get(&x)
            .unwrap_or_else(|| raise_unbound_identifier(x))
    }

    fn set_curr_lbl(&self, lbl: &'a str) -> Ctxt<'a> {
        Ctxt {
            curr_lbl: Some(lbl),
            ..self.clone()
        }
    }

    fn next_local(&self) -> (Ctxt<'a>, MemRef) {
        let si: i32 = (self.si + 1).try_into().unwrap();
        (
            Ctxt {
                si: self.si + 1,
                ..self.clone()
            },
            mref![Rbp - %(8 * si)],
        )
    }

    fn add_binding(&self, x: Symbol, mem: MemRef) -> Ctxt<'a> {
        Ctxt {
            env: self.env.update(x, mem),
            ..*self
        }
    }
}

pub fn compile(prg: &Prog) -> String {
    match fun_arity_map(prg) {
        Ok(funs) => {
            let mut sess = Session::new(funs);
            let locals = depth(&prg.main);
            sess.compile_funs(&prg.funs);
            sess.emit_instr(Instr::Label("our_code_starts_here".to_string()));
            let callee_saved = [Rbp, INPUT_REG];
            sess.fun_entry(locals, &callee_saved);
            sess.emit_instrs([
                Instr::Mov(MovArgs::ToReg(STACK_BASE, Arg64::Reg(Rbp))),
                Instr::Mov(MovArgs::ToReg(INPUT_REG, Arg64::Reg(Rdi))),
            ]);
            sess.compile_expr(&Ctxt::new(), Loc::Reg(Rax), &prg.main);
            sess.fun_exit(locals, &callee_saved);
            format!(
                "
section .text
extern snek_error
extern snek_print
extern snek_print_stack
global our_code_starts_here
{}
{INVALID_ARG}:
  mov edi, 1
  call snek_error
{OVERFLOW}:
  mov edi, 2
  call snek_error
",
                instrs_to_string(&sess.instrs)
            )
        }
        Err(dup) => raise_duplicate_function(dup),
    }
}

impl Session {
    fn new(funs: HashMap<Symbol, usize>) -> Session {
        Session {
            tag: 0,
            instrs: vec![],
            funs,
        }
    }

    fn fun_entry(&mut self, locals: u32, callee_saved: &[Reg]) {
        let size = frame_size(locals, callee_saved);
        for reg in callee_saved {
            self.emit_instr(Instr::Push(Arg32::Reg(*reg)));
        }
        self.emit_instrs([
            Instr::Mov(MovArgs::ToReg(Rbp, Arg64::Reg(Rsp))),
            Instr::Sub(BinArgs::ToReg(Rsp, Arg32::Imm(8 * (size as i32)))),
        ]);
    }

    fn fun_exit(&mut self, locals: u32, calle_saved: &[Reg]) {
        let size = frame_size(locals, calle_saved);
        self.emit_instrs([Instr::Add(BinArgs::ToReg(
            Rsp,
            Arg32::Imm(8 * (size as i32)),
        ))]);
        for reg in calle_saved.iter().rev() {
            self.emit_instr(Instr::Pop(Loc::Reg(*reg)));
        }
        self.emit_instr(Instr::Ret);
    }

    fn compile_funs(&mut self, funs: &[FunDecl]) {
        for fun in funs {
            self.compile_fun(fun)
        }
    }

    fn compile_fun(&mut self, fun: &FunDecl) {
        check_dup_bindings(&fun.params);
        let locals = depth(&fun.body);
        let mut cx = Ctxt::with_params(&fun.params);
        cx.tail_pos = true;
        self.emit_instr(Instr::Label(fun_label(fun.name)));
        self.fun_entry(locals, &[Rbp]);
        self.compile_expr(&cx, Loc::Reg(Rax), &fun.body);
        self.fun_exit(locals, &[Rbp]);
    }

    fn compile_expr(&mut self, cx: &Ctxt, dst: Loc, e: &Expr) {
        println!("compile_expr: expr - {:?} \ntail_pos {}", e, cx.tail_pos);
        match e {
            Expr::Number(n) => self.move_to(dst, n.repr64()),
            Expr::Boolean(b) => self.move_to(dst, b.repr64()),
            Expr::Var(x) => self.move_to(dst, Arg32::Mem(cx.lookup(*x))),
            Expr::Let(bindings, body) => {
                check_dup_bindings(bindings.iter().map(|(id, _)| id));
                let mut currcx = cx.clone();
                currcx.tail_pos = false;
                for (var, rhs) in bindings {
                    let (nextcx, mem) = currcx.next_local();
                    self.compile_expr(&currcx, Loc::Mem(mem), rhs);
                    currcx = nextcx.add_binding(*var, mem);
                }
                self.compile_expr(&currcx, Loc::Reg(Rax), body);
                self.move_to(dst, Arg64::Reg(Rax))
            },
            Expr::UnOp(op, e) => self.compile_un_op(cx, dst, *op, e),
            Expr::BinOp(op, e1, e2) => self.compile_bin_op(cx, dst, *op, e1, e2),
            Expr::If(e1, e2, e3) => {
                let tag = self.next_tag();
                let else_lbl = format!("if_else_{tag}");
                let end_lbl = format!("if_end_{tag}");

                self.compile_expr(&cx.with_tail_pos(false), Loc::Reg(Rax), e1);
                self.emit_instrs([
                    Instr::Cmp(BinArgs::ToReg(Rax, false.repr32().into())),
                    Instr::Je(else_lbl.clone()),
                ]);
                //println!("Inside If tail: {}", cx.tail_pos);
                self.compile_expr(cx, dst, e2);
                self.emit_instrs([Instr::Jmp(end_lbl.clone()), Instr::Label(else_lbl)]);
                self.compile_expr(cx, dst, e3);
                self.emit_instr(Instr::Label(end_lbl))
            },
            Expr::Loop(e) => {
                let tag = self.next_tag();
                let loop_start_lbl = format!("loop_start_{tag}");
                let loop_end_lbl = format!("loop_end_{tag}");

                self.emit_instr(Instr::Label(loop_start_lbl.clone()));
                // In a loop, the loop body is never in tail position, but the breaks inside it are if the loop is in tail position.
                self.compile_expr(&cx.set_curr_lbl(&loop_end_lbl).with_tail_pos_and_loop(false, cx.tail_pos), dst, e);
                self.emit_instrs([Instr::Jmp(loop_start_lbl), Instr::Label(loop_end_lbl)]);
                self.move_to(dst, Arg64::Reg(Rax));
            },
            Expr::Break(e) => {
                if let Some(lbl) = cx.curr_lbl {
                    self.compile_expr(&cx.with_tail_pos(cx.in_tail_loop), Loc::Reg(Rax), e);
                    self.emit_instr(Instr::Jmp(lbl.to_string()));
                } else {
                    raise_break_outside_loop()
                }
            },
            Expr::Set(var, e) => {
                let mem = cx.lookup(*var);
                self.compile_expr(&cx.with_tail_pos(false), Loc::Mem(mem), e);
                self.move_to(dst, Arg32::Mem(mem));
            },
            Expr::Block(es) => {
                for e in &es[..es.len() - 1] {
                    self.compile_expr(&cx.with_tail_pos(false), Loc::Reg(Rcx), e);
                }
                self.compile_expr(cx, dst, &es[es.len() - 1]);
            },
            Expr::Call(fun, args) => {
                let Some(arity) = self.funs.get(fun) else {
                    return raise_undefined_fun(*fun);
                };
                if args.len() != *arity {
                    raise_wrong_number_of_args(*fun, *arity, args.len());
                }

                let mut currcx = cx.with_tail_pos(false);
                for arg in args.iter().rev() {
                    let (nextcx, mem) = currcx.next_local();
                    self.compile_expr(&currcx, Loc::Mem(mem), arg);
                    currcx = nextcx;
                }
                self.call(*fun, locals(cx.si, args.len() as u32).map(Arg32::Mem), cx.tail_pos);
                self.move_to(dst, Arg64::Reg(Rax));
            },
            Expr::Input => {
                if cx.in_fun {
                    raise_input_in_fun()
                } else {
                    self.move_to(dst, Arg32::Reg(INPUT_REG))
                }
            },
            Expr::PrintStack => {
                self.emit_instrs([
                    Instr::Mov(MovArgs::ToReg(Rdi, Arg64::Reg(STACK_BASE))),
                    Instr::Mov(MovArgs::ToReg(Rsi, Arg64::Reg(Rbp))),
                    Instr::Mov(MovArgs::ToReg(Rdx, Arg64::Reg(Rsp))),
                    Instr::Call("snek_print_stack".to_string()),
                ]);
                self.move_to(dst, 0.repr32());
            }
        }
    }

    fn call(&mut self, fun: Symbol, args: impl IntoIterator<Item = Arg32>, tail_pos: bool) {
        println!("calling {} tail: {}", fun, tail_pos);
        let args: Vec<_> = args.into_iter().collect();
        if tail_pos {
            for (i, arg) in args.iter().rev().enumerate() {
                self.emit_instrs([
                    Instr::Mov(MovArgs::ToReg(Rdx, arg32to64(*arg))),
                    Instr::Mov(MovArgs::ToMem(MemRef { reg: Rbp, offset: Offset::Constant(((i+2) * 8) as i32) }, Reg32::Reg(Rdx)))
                ])
            }
            self.emit_instrs([
                Instr::Mov(MovArgs::ToReg(Rsp, Arg64::Reg(Rbp))),
                Instr::Add(BinArgs::ToReg(Rsp, Arg32::Imm(8))),
                Instr::Mov(MovArgs::ToReg(Rbp, Arg64::Mem(MemRef { reg: Rbp, offset: Offset::Constant(0) }))),
                Instr::Jmp(fun_label(fun)),
            ]);
        } else {
            for arg in args.iter() {
                self.emit_instr(Instr::Push(*arg))
            }
            self.emit_instrs([
                Instr::Call(fun_label(fun)),
                Instr::Add(BinArgs::ToReg(Rsp, Arg32::Imm(8 * args.len() as i32))),
            ]);
        }





        // if tail_pos {
        //     /*
        //         Strategy
        //         1. Start filling from the bottom with the last argument
        //         2. If you can fit it in before the return address, do so
        //         3. If you can't, move the return address down and try again
        //         4. Set the RSP to be RBP + 8
        //         5. Jump to the function
        //      */
        //     // // rsp <- rbp + 8 (set rsp to ret addr)
        //     // self.emit_instrs([
        //     //     Instr::Mov(MovArgs::ToReg(Rsp, Arg64::Reg(Rbp))),
        //     //     Instr::Add(BinArgs::ToReg(Rsp, Arg32::Imm(8))),
        //     // ]);
        //     // let args: Vec<_> = args.into_iter().collect();
        //     // for (i, arg) in args.iter().enumerate() {
        //     //     self.emit_instr(Instr::Mov(MovArgs::ToReg(Rdx, arg32to64(*arg))));
        //     //     self.emit_instr(Instr::Mov(MovArgs::ToMem(MemRef { reg: Rsp, offset: Offset::Constant(((i as i32 + 2) * 8) as i32 ) }, Reg32::Reg(Rdx))));
        //     // }
        //     // self.emit_instr(Instr::Jmp(fun_label(fun)));
        //     let mut args: Vec<_> = args.into_iter().collect();

        //     for arg in args.iter() {
        //         self.emit_instr(Instr::Push(*arg))
        //     }
        //     if args.len() % 2 != 0 {
        //         args.push(Arg32::Imm(MEM_SET_VAL));
        //     }
        //     self.emit_instrs([
        //         Instr::Call(fun_label(fun)),
        //         Instr::Add(BinArgs::ToReg(Rsp, Arg32::Imm(8 * args.len() as i32))),
        //     ]);
        // } else {
        //     let mut args: Vec<_> = args.into_iter().collect();
        //     if args.len() % 2 != 0 {
        //         args.push(Arg32::Imm(MEM_SET_VAL));
        //     }
        //     for arg in args.iter() {
        //         self.emit_instr(Instr::Push(*arg))
        //     }
        //     self.emit_instrs([
        //         Instr::Call(fun_label(fun)),
        //         Instr::Add(BinArgs::ToReg(Rsp, Arg32::Imm(8 * args.len() as i32))),
        //     ]);
        // }
    }

    fn compile_un_op(&mut self, cx: &Ctxt, dst: Loc, op: Op1, e: &Expr) {
        let new_cx = cx.with_tail_pos(false);
        self.compile_expr(&new_cx, Loc::Reg(Rax), e);
        match op {
            Op1::Add1 => {
                self.check_is_num(Reg::Rax);
                self.emit_instrs([
                    Instr::Add(BinArgs::ToReg(Rax, 1.repr32())),
                    Instr::Jo(OVERFLOW.to_string()),
                ])
            }
            Op1::Sub1 => {
                self.check_is_num(Reg::Rax);
                self.emit_instrs([
                    Instr::Sub(BinArgs::ToReg(Rax, 1.repr32())),
                    Instr::Jo(OVERFLOW.to_string()),
                ])
            }
            Op1::IsNum => {
                self.emit_instrs([
                    Instr::And(BinArgs::ToReg(Rax, Arg32::Imm(0b001))),
                    Instr::Mov(MovArgs::ToReg(Rax, false.repr64())),
                    Instr::Mov(MovArgs::ToReg(Rcx, true.repr64())),
                    Instr::CMov(CMov::Z(Rax, Arg64::Reg(Rcx))),
                ]);
            }
            Op1::IsBool => {
                self.emit_instrs([
                    Instr::And(BinArgs::ToReg(Rax, Arg32::Imm(0b001))),
                    Instr::Cmp(BinArgs::ToReg(Rax, Arg32::Imm(0b001))),
                    Instr::Mov(MovArgs::ToReg(Rax, false.repr64())),
                    Instr::Mov(MovArgs::ToReg(Rcx, true.repr64())),
                    Instr::CMov(CMov::E(Rax, Arg64::Reg(Rcx))),
                ]);
            }
            Op1::Print => self.emit_instrs([
                Instr::Mov(MovArgs::ToReg(Rdi, Arg64::Reg(Rax))),
                Instr::Call("snek_print".to_string()),
            ]),
        }
        self.move_to(dst, Arg32::Reg(Rax));
    }

    fn compile_bin_op(&mut self, cx: &Ctxt, dst: Loc, op: Op2, e1: &Expr, e2: &Expr) {
        let new_cx = cx.with_tail_pos(false);
        let (nextcx, mem) = new_cx.next_local();
        self.compile_expr(&new_cx, Loc::Mem(mem), e1);
        self.compile_expr(&nextcx, Loc::Reg(Rcx), e2);
        self.emit_instr(Instr::Mov(MovArgs::ToReg(Rax, Arg64::Mem(mem))));

        match op {
            Op2::Plus
            | Op2::Minus
            | Op2::Times
            | Op2::Greater
            | Op2::GreaterEqual
            | Op2::Less
            | Op2::LessEqual => {
                self.check_is_num(Rax);
                self.check_is_num(Rcx);
            }
            Op2::Equal => {
                

                let tag = self.next_tag();
                let check_eq_finish_lbl = format!("check_eq_finish_{tag}");
                // if (%rax ^ %rcx) & 0b01 == 0 {
                //     jmp check_eq_finish
                // } else if (%rax | %rcx) & 0b01 != 0 {
                //     jmp invalid_arg
                // }
                self.emit_instrs([
                    Instr::Mov(MovArgs::ToReg(Rdx, Arg64::Reg(Rax))),
                    Instr::Xor(BinArgs::ToReg(Rdx, Arg32::Reg(Rcx))),
                    Instr::Test(BinArgs::ToReg(Rdx, Arg32::Imm(0b01))),
                    Instr::Jz(check_eq_finish_lbl.to_string()),
                    Instr::Mov(MovArgs::ToReg(Rdx, Arg64::Reg(Rax))),
                    Instr::Or(BinArgs::ToReg(Rdx, Arg32::Reg(Rcx))),
                    Instr::Test(BinArgs::ToReg(Rdx, Arg32::Imm(0b01))),
                    Instr::Jnz(INVALID_ARG.to_string()),
                    Instr::Label(check_eq_finish_lbl.to_string()),
                ]);
            }
        }

        match op {
            Op2::Plus => {
                self.emit_instrs([
                    Instr::Add(BinArgs::ToReg(Rax, Arg32::Reg(Rcx))),
                    Instr::Jo(OVERFLOW.to_string()),
                ]);
            }
            Op2::Minus => {
                self.emit_instrs([
                    Instr::Sub(BinArgs::ToReg(Rax, Arg32::Reg(Rcx))),
                    Instr::Jo(OVERFLOW.to_string()),
                ]);
            }
            Op2::Times => {
                self.emit_instrs([
                    Instr::Sar(BinArgs::ToReg(Rax, Arg32::Imm(1))),
                    Instr::IMul(BinArgs::ToReg(Rax, Arg32::Reg(Rcx))),
                    Instr::Jo(OVERFLOW.to_string()),
                ]);
            }
            Op2::Equal => self.compile_cmp(CMov::E),
            Op2::Greater => self.compile_cmp(CMov::G),
            Op2::GreaterEqual => self.compile_cmp(CMov::GE),
            Op2::Less => self.compile_cmp(CMov::L),
            Op2::LessEqual => self.compile_cmp(CMov::LE),
        }
        self.move_to(dst, Arg32::Reg(Rax));
    }

    fn compile_cmp(&mut self, cmp: impl FnOnce(Reg, Arg64) -> CMov) {
        self.emit_instrs([
            Instr::Cmp(BinArgs::ToReg(Rax, Arg32::Reg(Rcx))),
            Instr::Mov(MovArgs::ToReg(Rax, false.repr64())),
            Instr::Mov(MovArgs::ToReg(Rcx, true.repr64())),
            Instr::CMov(cmp(Rax, Arg64::Reg(Rcx))),
        ]);
    }

    fn move_to(&mut self, dst: Loc, src: impl Into<Arg64>) {
        let src = src.into();
        if dst == src {
            return;
        }
        match (dst, src) {
            (Loc::Reg(reg), _) => self.emit_instr(Instr::Mov(MovArgs::ToReg(reg, src))),
            (Loc::Mem(dst), Arg64::Reg(src)) => {
                self.emit_instr(Instr::Mov(MovArgs::ToMem(dst, Reg32::Reg(src))))
            }
            (Loc::Mem(dst), Arg64::Imm(src)) => {
                if let Ok(src) = src.try_into() {
                    self.emit_instr(Instr::Mov(MovArgs::ToMem(dst, Reg32::Imm(src))))
                } else {
                    self.emit_instrs([
                        Instr::Mov(MovArgs::ToReg(Rdx, Arg64::Imm(src))),
                        Instr::Mov(MovArgs::ToMem(dst, Reg32::Reg(Rdx))),
                    ])
                }
            }
            (Loc::Mem(dst), Arg64::Mem(src)) => self.emit_instrs([
                Instr::Mov(MovArgs::ToReg(Rdx, Arg64::Mem(src))),
                Instr::Mov(MovArgs::ToMem(dst, Reg32::Reg(Rdx))),
            ]),
        }
    }

    fn check_is_num(&mut self, reg: Reg) {
        if env::var("DEBUG").is_ok() {
            return;
        }
        self.emit_instrs([
            Instr::Test(BinArgs::ToReg(reg, Arg32::Imm(0b001))),
            Instr::Jnz(INVALID_ARG.to_string()),
        ]);
    }

    fn emit_instrs(&mut self, instrs: impl IntoIterator<Item = Instr>) {
        self.instrs.extend(instrs);
    }

    fn emit_instr(&mut self, instr: Instr) {
        self.instrs.push(instr)
    }

    fn next_tag(&mut self) -> u32 {
        self.tag = self.tag.checked_add(1).unwrap();
        self.tag - 1
    }
}

fn locals(start: u32, count: u32) -> impl Iterator<Item = MemRef> {
    (start..start + count).map(|i| mref![Rbp - %(8 * (i + 1))])
}

fn frame_size(locals: u32, calle_saved: &[Reg]) -> u32 {
    // #locals + #callee saved + return address
    let n = locals + calle_saved.len() as u32 + 1;
    if n % 2 == 0 {
        locals
    } else {
        locals + 1
    }
}

fn depth(e: &Expr) -> u32 {
    match e {
        Expr::BinOp(_, e1, e2) => depth(e1).max(depth(e2) + 1),
        Expr::Let(bindings, e) => bindings
            .iter()
            .enumerate()
            .map(|(i, (_, e))| depth(e) + (i as u32))
            .max()
            .unwrap_or(0)
            .max(depth(e) + bindings.len() as u32),
        Expr::If(e1, e2, e3) => depth(e1).max(depth(e2)).max(depth(e3)),
        Expr::Block(es) => es.iter().map(depth).max().unwrap_or(0),
        Expr::UnOp(_, e) | Expr::Loop(e) | Expr::Break(e) | Expr::Set(_, e) => depth(e),
        Expr::Call(_, es) => es
            .iter()
            .enumerate()
            .map(|(i, e)| depth(e) + (i as u32))
            .max()
            .unwrap_or(0)
            .max(es.len() as u32),
        Expr::Input
        | Expr::Var(_)
        | Expr::Number(_)
        | Expr::Boolean(_) 
        | Expr::PrintStack => 0
    }
}

trait Repr64 {
    fn repr64(&self) -> Arg64;
}

trait Repr32 {
    fn repr32(&self) -> Arg32;
}

impl<T: Repr32> Repr64 for T {
    fn repr64(&self) -> Arg64 {
        self.repr32().into()
    }
}

impl Repr32 for i32 {
    fn repr32(&self) -> Arg32 {
        Arg32::Imm(*self << 1)
    }
}

impl Repr64 for i64 {
    fn repr64(&self) -> Arg64 {
        Arg64::Imm(self.checked_shl(1).unwrap())
    }
}

impl Repr32 for bool {
    fn repr32(&self) -> Arg32 {
        Arg32::Imm(if *self { 3 } else { 1 })
    }
}

fn fun_arity_map(prg: &Prog) -> Result<HashMap<Symbol, usize>, Symbol> {
    let mut map = HashMap::new();
    for fun in &prg.funs {
        if map.insert(fun.name, fun.params.len()).is_some() {
            return Err(fun.name);
        }
    }
    Ok(map)
}

fn check_dup_bindings<'a>(bindings: impl IntoIterator<Item = &'a Symbol>) {
    let mut seen = HashSet::new();
    for name in bindings {
        if !seen.insert(*name) {
            raise_duplicate_binding(*name);
        }
    }
}

fn raise_duplicate_binding(id: Symbol) {
    panic!("duplicate binding {id}");
}

fn raise_duplicate_function<T>(name: Symbol) -> T {
    panic!("duplicate function name {name}")
}

fn raise_unbound_identifier<T>(id: Symbol) -> T {
    panic!("unbound variable identifier {id}")
}

fn raise_break_outside_loop() {
    panic!("break outside loop")
}

fn raise_input_in_fun<T>() -> T {
    panic!("cannot use input inside function definition")
}

fn raise_undefined_fun(fun: Symbol) {
    panic!("function {fun} not defined")
}

fn raise_wrong_number_of_args(fun: Symbol, expected: usize, got: usize) {
    panic!("function {fun} takes {expected} arguments but {got} were supplied")
}

fn fun_label(fun: Symbol) -> String {
    format!("snek_fun_{}", fun.replace("-", "_"))
}

fn arg32to64(arg: Arg32) -> Arg64 {
    match arg {
        Arg32::Reg(reg) => Arg64::Reg(reg),
        Arg32::Imm(imm) => Arg64::Imm(imm as i64),
        Arg32::Mem(mem) => Arg64::Mem(mem),
    }
}