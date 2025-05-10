# Grammar

# Expressions

note: ?, \*, and + follow regex rules

## Constant assignment 

ident : type? : const_value ;?

## Declaration

ident : type? ;

## Assignment

ident : type? = value ;

## Import

import module ;

desugars to:

```
_.* = import module;
```

# Syntax example:

```go
io :: import std.io;

Response :: struct {
	msg: string,
	etc: bool,
}

Example :: enum {
	One,
	Two,
	Three,
}

main :: () -> i32 {
	io.print("hello world!");
	input := io.read_line() else {
		abort("Failed to read from console");
	};

	array : [4]Type<i32> = [0, 1, 2, 3];

	example := Example.One;
	switch example {
		One => io.print("One"),
		Two => io.print("Two"),
		Three => io.print("Three"),
	}

	if input == "hi" {
		io.print("hey!");
	}
	else {
		io.printf("You said: {}, and I dont blame you!", input);
	}
}
```