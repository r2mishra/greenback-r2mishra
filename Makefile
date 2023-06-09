UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
ARCH := elf64
endif
ifeq ($(UNAME), Darwin)
ARCH := macho64
TARGET := --target x86_64-apple-darwin
endif

tests/%.s: tests/%.snek src/main.rs
	cargo run -- $< tests/$*.s

tests/%.run: tests/%.s runtime/start.rs
	nasm -f $(ARCH) tests/$*.s -o tests/$*.o
	ar rcs tests/lib$*.a tests/$*.o
	rustc $(TARGET) -g -L tests/ -lour_code:$* runtime/start.rs -o tests/$*.run

.PHONY: test
test:
	cargo build
	cargo test

clean:
	rm -f tests/*.a tests/*.s tests/*.run tests/*.o
	rm -f runtime/libour_code.a runtime/our_code.o tests/out.run
	rm -f -r tests/*.dSYM

check: runtime/start.rs
	@rm -f runtime/libour_code.a runtime/our_code.o tests/out.run
	cargo run -- tests/in.snek tests/out.s
	nasm -f macho64 tests/out.s -o runtime/our_code.o
	ar rcs runtime/libour_code.a runtime/our_code.o
	rustc $(TARGET) -L runtime/ runtime/start.rs -o tests/out.run
	@echo -e "\nExpression"
	cat tests/in.snek
	@echo -e "\nGenerated Assembly"
	cat tests/out.s
	@echo -e "\nResult"
	./tests/out.run $(input)


ass : runtime/start.rs
	@rm -f runtime/libour_code.a runtime/our_code.o tests/out.run
	nasm -f macho64 tests/out.s -o runtime/our_code.o
	ar rcs runtime/libour_code.a runtime/our_code.o
	rustc $(TARGET) -L runtime/ runtime/start.rs -o tests/out.run
	@echo -e "\nExpression"
	# cat tests/in.snek
	@echo -e "\nGenerated Assembly"
	# cat tests/out.s
	@echo -e "\nResult"
	./tests/out.run $(input)

