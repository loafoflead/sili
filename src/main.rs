use std::io;
use std::collections::HashMap;
use std::fmt::{self, Display};

const PUNCTS: &[char] = &[',', ';', '.'];
const KEYWORDS: &[&str] = &["exit", "print"];

trait Optionalise where Self: Sized {
    fn some(self) -> Option<Self> {
        Some(self)
    }
}

impl<T> Optionalise for T {}

fn read_line() -> Result<String, io::Error> {
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    return Ok(buf);
}

#[derive(Debug)]
enum MachineError {

}

impl Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "")
    }
}

struct Machine {
    ints: HashMap<String, i32>,
}

impl Machine {
    fn new() -> Self {
        return Self {
            ints: HashMap::new(),
        }
    }

    /// Read some code and try running it
    fn understand(&mut self, code: impl ToString) -> Result<(), MachineError> {
        let ast = Ast {
            puncts: PUNCTS,
            string_enter: '<',
            string_exit: '>',
            keywords: KEYWORDS,
        };
        let tokens = ast.parse(code).expect("Syntax error");

        for token in tokens {
            use Token as T;
            match token {
                T::Ident(ident) => {
                    println!("Ident {ident} found.");
                }
                tok => todo!("Implement {tok:?}")
            }            
        }

        Ok(())
    }
}

struct Ast {
    puncts:         &'static [char],
    keywords:       &'static [&'static str],
    string_enter:   char,
    string_exit:    char,
}

impl Ast {
    fn parse(&self, text: impl ToString) -> Option<Vec<Token>> {
        let string = text.to_string();
        let chars = string.chars().collect::<Vec<char>>();

        let mut tokens: Vec<Token> = vec![];

        let mut token: Option<Token> = None;
        let mut buf = String::with_capacity(10);
        let mut in_string = false;

        for c in chars {
            if in_string {
                if c == self.string_exit {
                    in_string = false;
                    tokens.push(Token::StringLiteral(buf.drain(..).collect::<_>()));
                }
                else {
                    buf.push(c);
                }
            }
            else if c.is_whitespace() && !buf.trim().is_empty() {
                tokens.push(Token::from(buf.as_str().trim())?);
                buf.clear();
            }
            else if c == self.string_enter {
                in_string = true;
            }
            else if self.puncts.contains(&c) {
                if !buf.trim().is_empty() {
                    tokens.push(Token::from(buf.as_str().trim())?);
                    buf.clear();
                }
                tokens.push(Token::Punct(c));
            }
            else {
                buf.push(c);
            }
        }

        tokens = tokens.into_iter().map(|t| {
            if let Token::Ident(ref ident) = t { 
                if self.keywords.contains(&ident.as_str()) { Token::Keyword(ident.to_string()) } 
                else { t }
            }
            else { t }
        }).collect();

        tokens.some()
    }
}

#[derive(Debug)]
enum Token {
    IntLiteral(i32),
    FloatLiteral(f32),
    StringLiteral(String),
    Ident(String),
    Keyword(String),
    Punct(char),
}

impl Token {
    fn from(text: &str) -> Option<Self> {
        if let Ok(val) = text.parse::<i32>() {
            return Self::IntLiteral(val).some();
        }
        else if let Ok(val) = text.parse::<f32>() {
            return Self::FloatLiteral(val).some();
        }
        else {
            return Self::Ident(text.to_string()).some();
        }
    }
}

fn main() {
    let mut machine = Machine::new();

    loop {
        let input = read_line().expect("Realistically a really bad error.");

        if let Err(_) = machine.understand(input) {
            println!("Error reading input!");
        }
    }
}
