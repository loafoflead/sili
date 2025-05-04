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
import std->io;

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
	print("hello world!");
	input := read_line() else {
		panic("Failed to read from console");
	};

	example := Example->One;
	switch example {
		One => print("One"),
		Two => print("Two"),
		Three => print("Three"),
	}

	if input == "hi" {
		print("hey!");
	}
	else {
		print("You said: {}, and I dont blame you!", input);
	}
}
```