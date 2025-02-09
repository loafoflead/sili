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
    CannotWriteTo(Token),
    CannotWriteToMemory(Memory),
    CannotReadFrom(Token),
    InvalidDest(Token),
    UnkownRegister(String),
    ParsingError,
}

impl Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::SyntaxError(loc) => write!(f, "{loc}: Syntax error..."),
            Self::UnkownRegister(reg) => write!(f, "(?TODO?): Unknown register: {reg}"),
            Self::CannotWriteTo(token) => write!(f, "{loc}: Invalid write operation destination, expected register or memory; found: {:?}", token.ty, loc = token.loc),
            Self::CannotWriteToMemory(mem) => write!(f, "(?TODO?): Invalid write operation destination, expected register or pointer; found: {:?}", mem.ty),
            Self::CannotReadFrom(token) => write!(f, "{loc}: Invalid read operation source, expected register, memory, or constant; found: {:?}", token.ty, loc = token.loc),
            Self::InvalidDest(token) => write!(f, "{loc}: Invalid operation destination: {:?}", token.ty, loc = token.loc),
            Self::ParsingError => todo!("Implement parsing error handling."),
        }
    }
}

#[derive(Debug, Clone)]
struct Memory {
    ty: MemoryType,
}

impl Memory {
    fn is_writable(&self) -> bool {
        match self.ty {
            MemoryType::Reg(_) | MemoryType::Ptr(_) => true,
            MemoryType::UnsizedConstant(_) => false,
        }
    }

    fn reg(name: impl ToString) -> Self {
        Self {
            ty: MemoryType::Reg(name.to_string())
        }
    }

    fn usz(usz: usize) -> Self {
        Self {
            ty: MemoryType::UnsizedConstant(usz)
        }
    }
}

#[derive(Debug, Clone)]
enum MemoryType {
    Reg(String),
    Ptr(usize),
    UnsizedConstant(usize),
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

    fn to_writable(&self, token: &Token, deref: bool) -> Result<Memory, MachineError> {
        match &token.ty {
            TokenType::Ident(name) => {
                let _ptr = self.get_reg(name.as_str()).ok_or_else(|| todo!("Implement constants."))?;
                if deref { todo!("Implement dereferencing ptrs."); }
                // just a matter of using the value of get_reg i guess.

                Ok(Memory::reg(name.as_str()))
            }
            _ => Err(MachineError::CannotWriteTo(token.clone()))
        }
    }

    fn to_readable(&self, token: &Token, deref: bool) -> Result<Memory, MachineError> {
        match &token.ty {
            TokenType::Ident(name) => {
                let _ptr = self.get_reg(name.as_str()).ok_or_else(|| todo!("Implement constants."))?;
                if deref { todo!("Implement dereferencing ptrs."); }

                Ok(Memory::reg(name.as_str()))
            }
            TokenType::IntLiteral(int) => {
                Ok(Memory::usz(*int as usize))
            }
            _ => Err(MachineError::CannotReadFrom(token.clone()))
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

    // https://doc.rust-lang.org/std/mem/fn.transmute.html
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
        let tokens = ast.parse(code).ok_or(MachineError::ParsingError)?;
        let mut bytecode: Vec<u8> = vec![];

        let mut i = 0;

        while let Some(Token {ty, loc}) = tokens.get(i) {
            use TokenType as T;
            match ty {
                T::Keyword(kw) => {
                    i += 1;
                    match kw {
                        AsmKeyword::Mov => todo!("mov"),
                        // Accepted syntax:
                        // add reg const|reg|[ptr] (add rin 10) -- 2 or 4 tokens
                        // add [ptr] const|reg|[ptr] (deref)    -- 4 or 6 token
                        AsmKeyword::Add => {
                            let tokens_taken = tokens[i..]
                                .iter()
                                .position(|t| t.ty == TokenType::Newline || t.ty == TokenType::Eof)
                                .expect("Infinite input error! :O"); // this shouldn't ever fail

                            dbg!(i, tokens_taken);

                            if tokens_taken < 2 || tokens_taken == 3 || tokens_taken == 5 {
                                // TODO: TooFewArgs
                                return Err(MachineError::SyntaxError(loc.clone()));
                            }

                            let (dest, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                todo!("Implement dereferencing pointers");
                                // (dest, 3)
                            }
                            else if let Ok(writable) = self.to_writable(&tokens[i], false) {
                                (writable, 1)
                            } else {
                                return Err(MachineError::SyntaxError(loc.clone()));
                            };

                            i += n;

                            let (src, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                todo!("Implement dereferencing pointers");
                                // (dest, 3)
                            }
                            else if let Ok(readable) = self.to_readable(&tokens[i], false) {
                                (readable, 1)
                            } else {
                                return Err(MachineError::SyntaxError(loc.clone()));
                            };
                            
                            i += n;

                            let cur = self.read_mem(&dest);
                            let to_add = self.read_mem(&src);
                            self.write(&dest, &Memory::usz(cur + to_add))?;
                        }
                        AsmKeyword::Ret => {
                            bytecode.push(kw.to_bytecode());
                        }
                    }
                }
                T::Newline => i += 1,
                T::Eof => break,
                tok => todo!("Implement {tok:?}")
            }
        }

        Ok(())
    }

    fn read_mem(&self, mem: &Memory) -> usize {
        use MemoryType as Mt;
        match &mem.ty {
            Mt::Reg(name) => self.get_reg(name.as_str()).expect("Memory struct was invalid."),
            Mt::Ptr(_addr) => todo!("Read ptr"),
            Mt::UnsizedConstant(cons) => *cons
        }
    }

    fn write(&mut self, dest: &Memory, src: &Memory) -> Result<(), MachineError> {
        use MemoryType as Mt;
        if !dest.is_writable() {
            return Err(MachineError::CannotWriteToMemory(dest.clone()));
        }
        match (&dest.ty, &src.ty) {
            (Mt::Reg(name), Mt::UnsizedConstant(int)) => {
                self.set_reg(name.as_str(), *int as usize);
            }
            (Mt::Reg(name_dest), Mt::Reg(name_src)) => {
                let src = self.get_reg(name_src.as_str()).ok_or(MachineError::UnkownRegister(name_src.clone()))?;

                self.set_reg(name_dest.as_str(), src);
            }
            (Mt::Reg(_name_dest), Mt::Ptr(_src_addr)) => {
                todo!("Writing to register from memory");
            }
            // writing to memory
            (Mt::Ptr(_dest_addr), Mt::UnsizedConstant(_int)) => {
                todo!("Writing to memory");
            }
            (Mt::Ptr(_dest_addr), Mt::Reg(_name)) => {
                todo!("Writing from register to memory");
            }
            (Mt::Ptr(_dest_addr), Mt::Ptr(_src_addr)) => {
                todo!("Writing from memory to memory");
            }
            _ => unreachable!("is_writable check should prevent this.")
        }
        Ok(())
    }

    fn read_byte(&self, loc: usize) -> Option<u8> {
        if loc >= N {
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
        let mut column = 0;

        let mut tokens: Vec<Token> = vec![];

        let mut buf = String::with_capacity(10);
        let mut in_string = false;

        let mut evaluate_char = |c| {
            if in_string {
                if c == self.string_exit {
                    in_string = false;
                    tokens.push(Token::ty(TokenType::StringLiteral(buf.drain(..).collect::<_>()), line, column));
                }
                else {
                    buf.push(c);
                }
            }
            else if c.is_whitespace() && !buf.trim().is_empty() {
                if c == '\n' {
                    // TODO: extract this horrible logic :( i don't like repeating it
                    if !buf.trim().is_empty() {
                        tokens.push(Token::new(buf.as_str().trim(), line, column));
                        buf.clear();
                    }
                    tokens.push(Token::new("\n", line, column));
                    line += 1;
                    column = 0;
                }
                else {
                    tokens.push(Token::new(buf.as_str().trim(), line, column));
                }
                buf.clear();
            }
            else if c == self.string_enter {
                in_string = true;
            }
            else if self.puncts.contains(&c) {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line, column));
                    buf.clear();
                }
                tokens.push(Token::ty(TokenType::Punct(c), line, column));
            }
            else if c == '\0' {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line, column));
                    buf.clear();
                }
                tokens.push(Token::ty(TokenType::Eof, line, column));
            }
            else {
                buf.push(c);
            }
            column += 1;
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
    fn new(text: &str, line: usize, column: usize) -> Self {
        Self {
            ty: TokenType::from_str(text),
            loc: Location::new(line, column),
        }
    }

    fn ty(ty: TokenType, line: usize, column: usize) -> Self {
        Self {
            ty,
            loc: Location::new(line, column),
        }
    }
}

#[derive(Debug, Clone)]
struct Location {
    line: usize,
    column: usize,
}

impl Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}:{}", self.line, self.column)
    }
}

impl Location {
    fn new(line: usize, column: usize) -> Self {
        Self { line, column }
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
        let reg = "rin";
        let input = "add rout 12\nadd rin rout\nret";//read_line().expect("Realistically a really bad error.");

        println!("Register at {reg}: {byte}", byte=machine.get_reg(reg).expect("Tried to read unknown register"));

        if let Err(e) = machine.assemble(input) {
            println!("Error reading input: {}", e);
        }

        println!("Running the following code: \n{input}");
        
        let read_loc = 0x0;
        println!("Byte at {read_loc:#x}: {byte:#x}", byte=machine.read_byte(read_loc).expect("Out of bounds memory access."));
        println!("Register at {reg}: {byte}", byte=machine.get_reg(reg).expect("Tried to read unknown register"));
    // }
}
