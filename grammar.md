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
import std.io;

Response :: struct {
	msg: str,
	etc: bool,
}

Example :: enum {
	One,
	Two,
	Three,
}

Default :: trait {
	default :: fn() -> Self;
}

Example.* :: impl(Default) {
	default :: fn() -> Self {
		.One
	}
}

Example.* :: impl {
	next :: fn(&self) -> Self {
		switch self {
			.One   => .Two,
			.Two   => .Three,
			.Three => .One,
		}
	}
}

main :: () -> i32 {
	io.print("hello world!");
	input := io.read_line() else {
		panic("Failed to read from console");
	};

	if input == "hi" {
		io.print("hey!");
	}
	else {
		io.print(f"You said: {input}, and I dont blame you!");
	}
}
```