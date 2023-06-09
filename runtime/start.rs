use std::env;

type SnekVal = u64;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum ErrCode {
    InvalidArgument = 1,
    Overflow = 2
}

const TRUE: u64 = 3;
const FALSE: u64 = 1;

#[link(name = "our_code")]
extern "C" {
    // The \x01 here is an undocumented feature of LLVM that ensures
    // it does not add an underscore in front of the name.
    // Courtesy of Max New (https://maxsnew.com/teaching/eecs-483-fa22/hw_adder_assignment.html)
    #[link_name = "\x01our_code_starts_here"]
    fn our_code_starts_here(input: u64) -> u64;
}

#[export_name = "\x01snek_error"]
pub extern "C" fn snek_error(errcode: i64) {
    if errcode == ErrCode::InvalidArgument as i64 {
        eprintln!("invalid argument");
    } else if errcode == ErrCode::Overflow as i64 {
        eprintln!("overflow");
    } else {
        eprintln!("an error ocurred {}", errcode);
    }
    std::process::exit(errcode as i32);
}

#[export_name = "\x01snek_print"]
pub unsafe extern "C" fn snek_print(val: SnekVal) -> SnekVal {
    println!("{}", snek_str(val));
    val
}
unsafe fn snek_str(val: SnekVal) -> String {
    if val == TRUE {
        format!("true")
    } else if val == FALSE {
        format!("false")
    } else if val & 1 == 0 {
        format!("{}", (val as i64) >> 1)
    } else {
        format!("unknown value: {val}")
    }
}

#[export_name = "\x01snek_print_stack"]
pub unsafe extern "C" fn snek_print_stack(
    stack_base: *const u64,
    _curr_rbp: *const u64,
    curr_rsp: *const u64,
) {
    let mut ptr = stack_base;
    println!("-----------------------------------------");
    while ptr >= curr_rsp {
        let val = *ptr;
        if ptr == _curr_rbp {
            println!("{ptr:?}: {:#0x} [RBP]", val);
        } else {
            println!("{ptr:?}: {:#0x}", val);
        }
        ptr = ptr.sub(1);
    }
    println!("-----------------------------------------");
}

fn parse_input(input: &str) -> u64 {
    match input {
        "true" => TRUE,
        "false" => FALSE,
        _ => (input.parse::<i64>().unwrap() << 1) as u64,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = if args.len() >= 2 { &args[1] } else { "false" };
    let input = parse_input(&input);
    let i: u64 = unsafe { our_code_starts_here(input) };
    unsafe { snek_print(i) };
}
