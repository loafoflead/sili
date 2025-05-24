// use std::io;
// use std::fmt::{self, Display};
// use std::mem;

mod syn;
mod tokeniser;
mod type_check;

const PUNCTS: &[&str] = &["(", ")", "{", "}", ";", ":", "=", ",", ".", "->", "<", ">", "-", "+", "{]*=*["];
const KWORDS: &[&str] = &["struct", "enum", "if", "else", "return", "ext"];

fn create_tokeniser() -> tokeniser::Tokeniser {
    tokeniser::Tokeniser {
        puncts: PUNCTS,
        keywords: KWORDS,
        string_enter: '"',
        string_exit: '"',
        keep_newlines: false,
    }
}

#[test] 
fn basic() {
    let tokeniser = create_tokeniser();

    let snippet = 
r#"
value :: 5;
name :: "hello";
boolean_f :: false;
boolean_t :: true;

main :: () -> i32 {
    a :: 5;
    b :: 69420;
}
"#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    assert!(syn::parse_items(snippet, tokens).is_some());
}

#[test] 
fn function_arguments_and_fields() {
    let tokeniser = create_tokeniser();

    let snippet = 
r#"
foo :: (input: string, height: f32) {}

main :: () -> string {
    a: i32 : 5;
    foo("hello", 7);
    return 5;
}

Thing :: struct {
    height: i32,
    age: string,
    thing: bool,
}
"#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    assert!(syn::parse_items(snippet, tokens).is_some());
}

fn main() {
    let tokeniser = create_tokeniser();

    let snippet = 
r#"
Foo :: struct {
    height: Foo,
    age: i32,
}

main :: (bar: Foo) -> i32 {
    which :: 1;
    a :: which;
    return a;
}
"#;
// r#"
// main :: () -> i32 {
//     a : i32 : 5;
//     b := 5;
//     return a;
// }
// "#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    let items = syn::parse_items(snippet, tokens).unwrap();

    dbg!(&items);

    let _correct = type_check::type_check(items).unwrap();
}

/*
fn main() {
    language::Parser::parse(
// main : fn() : {
//     print 'hi';
// }
// main :: fn() {
"
    CONST :: 12303;
    CONST :: 12303;
VAL :int: 122;
"
// }
// Expr::Declaration {
//   ident: "main",
//   ty: Type::Function { args: [], returns: Unit },
//   value: Value::Expr {
//       expr: Expr::Block {..},
//   } 
// }
// Expr::Declaration {
//   ident: "val",
//   ty: Type::IntLiteral,
//   value: Value::Int(1),
// }
// "main :: fn() {
//     print 'hi';
// }"
).unwrap();
}
*/
