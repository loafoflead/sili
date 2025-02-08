use std::io;
use std::fmt::{self, Display};

const PUNCTS: &[char] = &[',', ';', '.'];
const CALL_STACK_SIZE: usize = 10;
const INSTRUCTION_START: usize = 0;

#[derive(Debug, PartialEq, Clone, Copy)]
enum AsmKeyword {
    /// mov dst, src: moves src (register, memory contents,
    /// or const) into dst (register or memory)
    Mov, 
    /// add dst, src: adds src (register, memory, const)
    /// into dst (register, memory)
    ///
    /// add [r1], r2: adds the value of r2 into the dereferenced value of r1
    /// add r1, r2  : adds the value of r2 to the pointer r1 (or just int)
    Add, 
    /// ret
    /// jumps back into the call stack
    Ret,
    // dat name, value: adds value into memory with the name [name]
    // Dat,
}

impl AsmKeyword {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "add" => Self::Add,
            "mov" => Self::Mov,
            "ret" => Self::Ret,
            _ => return None,
        }.some()
    }

    fn to_bytecode(&self) -> u8 {
        match self {
            Self::Mov => 1,
            Self::Add => 2,
            Self::Ret => 3,
        }
    }
}

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
    SyntaxError(Location),
    InvalidDest(Token),
}

impl Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::SyntaxError(loc) => write!(f, "Syntax error at: {loc}"),
            Self::InvalidDest(token) => write!(f, "{loc}: Invalid operation destination: {:?}", token.ty, loc = token.loc),
        }
    }
}

// Virtual machine ish thing
struct Machine<const N: usize> {
    memory: [u8; N],
    call_stack: [usize; CALL_STACK_SIZE],
    call_depth: usize,
    /// Instruction ptr
    rip: usize, 
    rin: usize,
    rout: usize,
    r1: usize,
    r2: usize,
}

impl<const N: usize> Machine<N> {
    fn new() -> Self {
        let mut call_stack = [0usize; CALL_STACK_SIZE];
        call_stack[0] = INSTRUCTION_START;
        return Self {
            memory: [0u8; N],
            call_stack,
            call_depth: 0,
            rip: INSTRUCTION_START, rin: 0, rout: 0, r1: 0, r2: 0,
        }
    }

    fn get_reg(&self, s: &str) -> Option<usize> {
        match s {
            "rip" => self.rip,
            "rin" => self.rin,
            "rout" => self.rout,
            "r1" => self.r1,
            "r2" => self.r2,
            _ => return None,
        }.some()
    }

    fn set_reg(&mut self, s: &str, val: usize) -> Option<()> {
        *(match s {
            "rip" => &mut self.rip,
            "rin" => &mut self.rin,
            "rout" => &mut self.rout,
            "r1" => &mut self.r1,
            "r2" => &mut self.r2,
            _ => return None,
        }) = val;
        return Some(());
    }

    fn assert_is_reg(&self, token: &Token) -> Result<(), MachineError> {
        let ident = if let TokenType::Ident(ident) = &token.ty { ident.as_str() } else {
            return Err(MachineError::InvalidDest(token.clone()));
        };

        if self.get_reg(ident).is_none() {
            return Err(MachineError::InvalidDest(token.clone()));
        }

        Ok(())
    }

    /// Read some assembly code and place it into memory
    fn assemble(&mut self, code: impl ToString) -> Result<(), MachineError> {
        let ast = Ast {
            puncts: PUNCTS,
            string_enter: '<',
            string_exit: '>',
        };
        let tokens = ast.parse(code).expect("Syntax error");
        let mut bytecode: Vec<u8> = vec![];

        dbg!(&tokens);

        let mut i = 0;

        while let Some(Token {ty, loc}) = tokens.get(i) {
            i += 1; // consume the instruction

            use TokenType as T;
            match ty {
                T::Keyword(kw) => {
                    println!("Keyword {kw:?} found.");
                    match kw {
                        AsmKeyword::Mov => todo!("mov"),
                        // Accepted syntax:
                        // add reg const|reg|[ptr] (add rin 10) -- 2 or 4 tokens
                        // add [ptr] const|reg|[ptr] (deref)    -- 4 or 6 token
                        AsmKeyword::Add => {
                            let newline = tokens
                                .iter()
                                .position(|t| t.ty == TokenType::Newline || t.ty == TokenType::Eof)
                                .expect("Infinite input error! :O"); // this shouldn't ever fail
                            let tokens_taken = newline - i;

                            dbg!(newline);
                            dbg!(&tokens[i..newline]);
                            dbg!(&tokens[i]);

                            if tokens_taken < 2 || tokens_taken == 3 || tokens_taken == 5 {
                                // TODO: TooFewArgs
                                return Err(MachineError::SyntaxError(loc.clone()));
                            }

                            let (dest, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                todo!("Implement dereferencing pointers");
                                // (dest, 3)
                            }
                            else {
                                self.assert_is_reg(&tokens[i])?;
                                (&tokens[i], 1)
                            };

                            i += n;

                            let (src, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                todo!("Implement dereferencing pointers");
                                // (dest, 3)
                            }
                            else if Ok(_) = self.assert_is_reg(&tokens[i]) {
                                (&tokens[i], 1)
                            }
                            else if let TokenType::IntLiteral(_) = &tokens[i] {
                                (&tokens[i], 1)
                            }
                            else {
                                todo!("Implement constants");
                            };
                            
                            i += n;

                            
                        }
                        AsmKeyword::Ret => {
                            bytecode.push(kw.to_bytecode());
                        }
                    }
                }
                tok => todo!("Implement {tok:?}")
            }
        }

        Ok(())
    }

    fn read_byte(&self, loc: usize) -> Option<u8> {
        if loc >= N || loc < 0 {
            return None;
        }
        self.memory[loc].some()
    }
}

struct Ast {
    puncts:         &'static [char],
    string_enter:   char,
    string_exit:    char,
}

impl Ast {
    fn parse(&self, text: impl ToString) -> Option<Vec<Token>> {
        let string = text.to_string();
        let chars = string.chars().collect::<Vec<char>>();

        let mut line = 0;

        let mut tokens: Vec<Token> = vec![];

        let mut buf = String::with_capacity(10);
        let mut in_string = false;

        let mut evaluate_char = |c| {
            if in_string {
                if c == self.string_exit {
                    in_string = false;
                    tokens.push(Token::ty(TokenType::StringLiteral(buf.drain(..).collect::<_>()), line));
                }
                else {
                    buf.push(c);
                }
            }
            else if c.is_whitespace() && !buf.trim().is_empty() {
                if c == '\n' {
                    // TODO: extract this horrible logic :( i don't like repeating it
                    if !buf.trim().is_empty() {
                        tokens.push(Token::new(buf.as_str().trim(), line));
                        buf.clear();
                    }
                    tokens.push(Token::new("\n", line));
                    line += 1;
                }
                else {
                    tokens.push(Token::new(buf.as_str().trim(), line));
                }
                buf.clear();
            }
            else if c == self.string_enter {
                in_string = true;
            }
            else if self.puncts.contains(&c) {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line));
                    buf.clear();
                }
                tokens.push(Token::ty(TokenType::Punct(c), line));
            }
            else if c == '\0' {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line));
                    buf.clear();
                }
                tokens.push(Token::ty(TokenType::Eof, line));
            }
            else {
                buf.push(c);
            }
        };

        for c in chars {
            evaluate_char(c);
        }
        evaluate_char('\0');

        tokens = tokens.into_iter().map(|t| {
            if let TokenType::Ident(ref ident) = t.ty { 
                if let Some(kwrd) = AsmKeyword::from_str(ident.as_str()) { Token { ty: TokenType::Keyword(kwrd), loc: t.loc } } 
                else { t }
            }
            else { t }
        }).collect();

        tokens.some()
    }
}

#[derive(Debug, Clone)]
struct Token {
    ty: TokenType,
    loc: Location,
}

impl Token {
    fn new(text: &str, line: usize) -> Self {
        Self {
            ty: TokenType::from_str(text),
            loc: Location::line(line),
        }
    }

    fn ty(ty: TokenType, line: usize) -> Self {
        Self {
            ty,
            loc: Location::line(line),
        }
    }
}

#[derive(Debug, Clone)]
struct Location {
    line: usize,
}

impl Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}:{}", self.line, "?")
    }
}

impl Location {
    fn line(line: usize) -> Self {
        Self { line }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    IntLiteral(i32),
    FloatLiteral(f32),
    StringLiteral(String),
    Ident(String),
    Keyword(AsmKeyword),
    Punct(char),
    Newline,
    Eof,
}

impl TokenType {
    fn from_str(text: &str) -> Self {
        if let Ok(val) = text.parse::<i32>() {
            Self::IntLiteral(val)
        }
        else if let Ok(val) = text.parse::<f32>() {
            Self::FloatLiteral(val)
        }
        else if text == "\n" {
            Self::Newline
        }
        else {
            Self::Ident(text.to_string())
        }
    }
}

fn main() {
    let mut machine = Machine::<128>::new();

    // add [rdi] 10 -> adds 10 to the area in memory the value of rdi is the address of
    // add rdi 10   -> adds 10 to the value of rdi

    // loop {
        let input = "add rin 10\nret";//read_line().expect("Realistically a really bad error.");

        if let Err(e) = machine.assemble(input) {
            println!("Error reading input: {}", e);
        }
        
        let read_loc = 0x0;
        println!("Byte at {read_loc:#x}: {byte:#x}", byte=machine.read_byte(read_loc).expect("Out of bounds memory access."));
    // }
}
