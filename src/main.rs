use std::io;
use std::fmt::{self, Display};
use std::mem;

const PUNCTS: &[char] = &[':', ';', '.', '[', ']'];
const CALL_STACK_SIZE: usize = 10;
const INSTRUCTION_START: usize = 0;

type Unit = usize;

const REGISTER_COUNT: Unit = 5;

const ADD_REG_USIZE: Unit = 5;
const ADD_REG_USIZE_END: Unit = ADD_REG_USIZE + REGISTER_COUNT;



// impl TryFrom<Unit> for Bytecode {
//     type Error = String;

//     fn try_from(u: Unit) -> Result<Self, Self::Error> {
//         if u > Self::AddMemToMem as Unit {
//             return Err(String::from("Invalid operation"));
//         }
//         Ok(u as Self)
//     }
// }

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
    SyntaxError(Location, Option<String>),
    CannotWriteTo(Token),
    CannotWriteToMemory(Memory),
    CannotReadFrom(Token),
    InvalidDest(Token),
    UnkownRegister(String),
    UnclosedParen(Token),
    OutOfBounds(usize),
    ParsingError,
    MachineCodeError,
}

impl Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::SyntaxError(loc, s) => write!(f, "{loc}: Syntax error: {msg}", msg = if s.is_some() { s.as_ref().unwrap() } else { "???" }),
            Self::UnkownRegister(reg) => write!(f, "(?TODO?): Unknown register: {reg}"),
            Self::OutOfBounds(addr) => write!(f, "(?TODO?): Out of bounds memory access: {addr}"),
            Self::CannotWriteTo(token) => write!(f, "{loc}: Invalid write operation destination, expected register or memory; found: {:?}", token.ty, loc = token.loc),
            Self::UnclosedParen(token) => write!(f, "{loc}: Unclosed parenthesis: {:?}", token.ty, loc = token.loc),
            Self::CannotWriteToMemory(mem) => write!(f, "(?TODO?): Invalid write operation destination, expected register or pointer; found: {:?}", mem.ty),
            Self::CannotReadFrom(token) => write!(f, "{loc}: Invalid read operation source, expected register, memory, or constant; found: {:?}", token.ty, loc = token.loc),
            Self::InvalidDest(token) => write!(f, "{loc}: Invalid operation destination: {:?}", token.ty, loc = token.loc),
            Self::ParsingError => todo!("Implement parsing error handling."),
            Self::MachineCodeError => write!(f, "'fraid you're a little screwed here;"),
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

    fn ptr(addr: usize) -> Self {
        Self {
            ty: MemoryType::Ptr(addr)
        }
    }
}

type MachineResult<T> = Result<T, MachineError>;

#[derive(Debug, Clone)]
enum MemoryType {
    Reg(String),
    Ptr(usize),
    UnsizedConstant(Unit),
}

#[derive(Debug, Clone, Copy)]
#[repr(usize)]
enum Register {
    Rzero = 0,
    Rone = 1,
    Rtwo = 2,
    Rip = 3,
    Rin = 4,
    Rout = 5,
}

impl From<Register> for usize {
    fn from(u: Register) -> Self {
        return u as usize;
    }
}

impl TryFrom<usize> for Register {
    type Error = MachineError;

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        Ok(match u {
             0 => Register::Rzero,
             1 => Register::Rone,
             2 => Register::Rtwo,
             3 => Register::Rip,
             4 => Register::Rin,
             5 => Register::Rout,
             _ => return Err(MachineError::UnkownRegister(format!("n°{u}"))),
        })
    }
}

impl Register {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "rip" => Self::Rip,
            "rin" => Self::Rin,
            "rout" => Self::Rout,
            "r0" => Self::Rzero,
            "r1" => Self::Rone,
            "r2" => Self::Rtwo,
            _ => return None,
        }.some()
    }

    fn bytecode(&self) -> usize {
        (*self).into()
    }
}

// Virtual machine ish thing
struct Machine<const N: usize> {
    memory: [u8; N],
    call_stack: [usize; CALL_STACK_SIZE],
    call_depth: usize,
    /// Instruction ptr
    rip: Unit, 
    rin: Unit,
    rout: Unit,
    r0: Unit,
    r1: Unit,
    r2: Unit,
}

impl<const N: usize> Machine<N> {
    fn new() -> Self {
        let mut call_stack = [0usize; CALL_STACK_SIZE];
        call_stack[0] = INSTRUCTION_START;
        return Self {
            memory: [0u8; N],
            call_stack,
            call_depth: 0,
            rip: INSTRUCTION_START, rin: 0, rout: 0, r0: 0, r1: 0, r2: 0,
        }
    }

    fn print_registers(&self) {
        println!(
            "Registers: r0: {}, r1: {}, r2: {}, rip: {}, rin: {}, rout: {}",
            self.r0,
            self.r1,
            self.r2,
            self.rip,
            self.rin,
            self.rout,
        );
    }

    fn to_writable(&self, token: &Token, deref: bool) -> Result<Memory, MachineError> {
        match &token.ty {
            TokenType::Ident(name) => {
                let _ptr = self.get_reg_value(name.as_str()).ok_or_else(|| todo!("Implement constants."))?;
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
                let _ptr = self.get_reg_value(name.as_str()).ok_or_else(|| todo!("Implement constants."))?;
                if deref { todo!("Implement dereferencing ptrs."); }

                Ok(Memory::reg(name.as_str()))
            }
            TokenType::IntLiteral(int) => {
                Ok(Memory::usz(*int as usize))
            }
            _ => Err(MachineError::CannotReadFrom(token.clone()))
        }
    }

    fn parse_deref(&self, tokens: &[Token]) -> Result<Memory, MachineError> {
        let Token { ty: TokenType::Ident(id), loc } = &tokens[0] else {
            let loc = &tokens[0].loc;
            return Err(MachineError::SyntaxError(loc.clone(), "Pointer arithmetic base must be a register, got a constant or keyword.".to_string().some()));
        };
        let Some(val) = self.get_reg_value(id.as_str()) else {
            return Err(MachineError::SyntaxError(loc.clone(), "Pointer arithmetic base must be a register.".to_string().some()));
        };

        if tokens.len() > 1 {
            todo!("Implement pointer arithmetic.");
        }

        return Ok(Memory::ptr(val));
    }

    fn reg_to_bytecode(&self, s: &str) -> Option<Unit> {
        let reg = Register::from_str(s)?;
        reg.bytecode().some()
    }

    fn get_reg_value(&self, s: &str) -> Option<usize> {
        let reg = Register::from_str(s)?;
        use Register as Reg;
        match reg {
            Reg::Rzero => self.r0,
            Reg::Rone => self.r1,
            Reg::Rtwo => self.r2,
            Reg::Rin => self.rin,
            Reg::Rout => self.rout,
            Reg::Rip => self.rip,
        }.some()
    }

    // https://doc.rust-lang.org/std/mem/fn.transmute.html
    fn set_reg(&mut self, s: &str, val: usize) -> Option<()> {
        let reg = Register::from_str(s)?;
        use Register as Reg;
        *(match reg {
            Reg::Rzero => &mut self.r0,
            Reg::Rone => &mut self.r1,
            Reg::Rtwo => &mut self.r2,
            Reg::Rin => &mut self.rin,
            Reg::Rout => &mut self.rout,
            Reg::Rip => &mut self.rip,
        }) = val;
        Some(())
    }

    fn set_reg_idx(&mut self, reg: usize, val: Unit) -> MachineResult<()> {
        use Register as Reg;
        *(match reg.try_into()? {
            Reg::Rzero => &mut self.r0,
            Reg::Rone => &mut self.r1,
            Reg::Rtwo => &mut self.r2,
            Reg::Rin => &mut self.rin,
            Reg::Rout => &mut self.rout,
            Reg::Rip => &mut self.rip,
        }) = val;
        Ok(())
    }

    fn get_reg_value_idx(&mut self, reg: usize) -> MachineResult<Unit> {
        use Register as Reg;
        Ok(match reg.try_into()? {
            Reg::Rzero => self.r0,
            Reg::Rone => self.r1,
            Reg::Rtwo => self.r2,
            Reg::Rin => self.rin,
            Reg::Rout => self.rout,
            Reg::Rip => self.rip,
        })
    }

    fn assert_is_reg(&self, token: &Token) -> Result<(), MachineError> {
        let ident = if let TokenType::Ident(ident) = &token.ty { ident.as_str() } else {
            return Err(MachineError::InvalidDest(token.clone()));
        };

        if self.get_reg_value(ident).is_none() {
            return Err(MachineError::InvalidDest(token.clone()));
        }

        Ok(())
    }

    /// Read some assembly code and place it into memory
    fn assemble(&mut self, code: impl ToString) -> Result<Vec<Unit>, MachineError> {
        let ast = Ast {
            puncts: PUNCTS,
            string_enter: '<',
            string_exit: '>',
        };
        let tokens = ast.parse(code).ok_or(MachineError::ParsingError)?;
        let mut bytecode: Vec<Unit> = vec![];

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

                            if tokens_taken < 2 || tokens_taken == 3 || tokens_taken == 5 {
                                // TODO: TooFewArgs
                                return Err(MachineError::SyntaxError(loc.clone(), None));
                            }

                            let (dest, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                i += 1;
                                let n = tokens[i..].iter().position(|t| t.ty == TokenType::Punct(']')).ok_or(MachineError::UnclosedParen(tokens[i].clone()))?;

                                if n > 1 {
                                    todo!("Implement pointer arithmetic when dereferencing.");
                                }

                                (self.parse_deref(&tokens[i..i+n])?, n+1) // +1 for the close paren
                            }
                            else if let Ok(writable) = self.to_writable(&tokens[i], false) {
                                (writable, 1)
                            } else {
                                return Err(MachineError::SyntaxError(loc.clone(), None));
                            };

                            i += n;

                            let (src, n) = if matches!(tokens[i].ty, TokenType::Punct('[')) {
                                i += 1;
                                let n = tokens[i..].iter().position(|t| t.ty == TokenType::Punct(']')).ok_or(MachineError::UnclosedParen(tokens[i].clone()))?;

                                if n > 1 {
                                    todo!("Implement pointer arithmetic when dereferencing.");
                                }

                                (self.parse_deref(&tokens[i..i+n])?, n+1) // +1 for the close paren
                            }
                            else if let Ok(readable) = self.to_readable(&tokens[i], false) {
                                (readable, 1)
                            } else {
                                return Err(MachineError::SyntaxError(loc.clone(), None));
                            };
                            
                            i += n;

                            bytecode.append(&mut self.add_args_to_bytecode(dest, src)?);
                            // let cur = self.read_mem(&dest)?;
                            // let to_add = self.read_mem(&src)?;
                            // self.write(&dest, &Memory::usz(cur + to_add))?;
                        }
                        AsmKeyword::Ret => ()
                    }
                }
                T::Ident(ident) => {
                    if tokens[i+1].ty == TokenType::Punct(':') {
                        todo!("Parse routine labels");
                    }
                }
                T::Newline => i += 1,
                T::Eof => break,
                tok => todo!("Implement {tok:?}")
            }
        }

        Ok(bytecode)
    }

    fn run_bytecode(&mut self, bytecode: &[Unit]) -> MachineResult<()> {
        let mut i = 0;
        while let Some(instruction) = bytecode.get(i) {
            let mut n = 1; // number of instructions consumed
            match instruction {
                0..ADD_REG_USIZE => todo!("Instructions 0-ADD"),

                ADD_REG_USIZE..=ADD_REG_USIZE_END => {
                    let reg = instruction - ADD_REG_USIZE;
                    let prev = self.get_reg_value_idx(reg as usize)?;
                    self.set_reg_idx(reg as usize, prev + *bytecode.get(i + 1).ok_or(MachineError::MachineCodeError)?)?;
                    n += 1;
                }

                val => todo!("Range {val} onwards..."),
            }
            i += n;
        }
        Ok(())
    }

    fn read_mem(&self, mem: &Memory) -> Result<usize, MachineError> {
        use MemoryType as Mt;
        Ok(match &mem.ty {
            Mt::Reg(name) => self.get_reg_value(name.as_str()).expect("Memory struct was invalid."),
            Mt::Ptr(addr) => self.read::<usize, {mem::size_of::<usize>()}>(*addr)?, // TODO: is this... right? 
            Mt::UnsizedConstant(cons) => *cons
        })
    }

    fn add_args_to_bytecode(&self, dest: Memory, src: Memory) -> Result<Vec<Unit>, MachineError> {
        use MemoryType as Mt;
        Ok(match (&dest.ty, &src.ty) {
            (Mt::Reg(name), Mt::UnsizedConstant(int)) => {
                vec![ADD_REG_USIZE + self.reg_to_bytecode(name).ok_or(MachineError::UnkownRegister(name.clone()))?, *int]
            }
            // (Mt::Reg(name_dest), Mt::Reg(name_src)) => {
            //     Bytecode::AddRegToReg(name_dest)
            // }
            // (Mt::Reg(name_dest), Mt::Ptr(src_addr)) => {
            //     Bytecode::AddMemToReg(name_dest)
            // }
            // // writing to memory
            // (Mt::Ptr(dest_addr), Mt::UnsizedConstant(int)) => {
            //     Bytecode::AddUsizeToMem
            // }
            // (Mt::Ptr(dest_addr), Mt::Reg(name_src)) => {
            //     Bytecode::AddRegToMem
            // }
            // (Mt::Ptr(dest_addr), Mt::Ptr(src_addr)) => {
            //     Bytecode::AddMemToMem
            // }
            _ => return Err(MachineError::CannotWriteToMemory(dest.clone())),
        })
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
                let src = self.get_reg_value(name_src.as_str())
                    .ok_or(MachineError::UnkownRegister(name_src.clone()))?;

                self.set_reg(name_dest.as_str(), src);
            }
            (Mt::Reg(name_dest), Mt::Ptr(src_addr)) => {
                let val = self.read::<usize, {mem::size_of::<usize>()}>(*src_addr)?;
                self.set_reg(name_dest.as_str(), val);
            }
            // writing to memory
            (Mt::Ptr(dest_addr), Mt::UnsizedConstant(int)) => {
                self.write_bytes(*dest_addr, &int.to_ne_bytes())?;
            }
            (Mt::Ptr(dest_addr), Mt::Reg(name_src)) => {
                let src = self.get_reg_value(name_src.as_str())
                    .ok_or(MachineError::UnkownRegister(name_src.clone()))?;

                self.write_bytes(*dest_addr, &src.to_ne_bytes())?;
            }
            (Mt::Ptr(dest_addr), Mt::Ptr(src_addr)) => {
                let val = self.read::<usize, {mem::size_of::<usize>()}>(*src_addr)?;
                self.write_bytes(*dest_addr, &val.to_ne_bytes())?;
            }
            _ => unreachable!("is_writable check should prevent this.")
        }
        Ok(())
    }

    fn write_bytes(&mut self, loc: usize, bytes: &[u8]) -> Result<(), MachineError> {
        if bytes.len() + loc >= N {
            return Err(MachineError::OutOfBounds(loc + bytes.len()));
        }

        // TODO: learn memcpy or whatever
        for (i, byte) in bytes.iter().enumerate() {
            self.memory[loc + i] = *byte;
        }
        Ok(())
    }

    // TODO: figure out what alignment is and how to not make this
    // code capable of crashing.
    fn read<T, const BYTES: usize>(&self, loc: usize) -> Result<T, MachineError> {
        let bytes = mem::size_of::<T>();

        if bytes > mem::size_of::<usize>() {
            todo!("Unimplemented, dereferencing larger than ptr size");
        }
        
        if loc + bytes >= N {
            return Err(MachineError::OutOfBounds(loc + bytes));
        }

        let memory: [u8; BYTES] = self.memory[loc..loc+bytes]
            .try_into()
            .map_err(
                |_| panic!("Invalid type size in read function, got {}, but needed {}", BYTES, mem::size_of::<T>())
            ).unwrap();

        // https://users.rust-lang.org/t/transmute-doesnt-work-on-generic-types/87272
        // FIXME: this is scary, i'd like to understand it l8r pls ty
        let val = unsafe { mem::transmute_copy::<[u8; BYTES], T>(&memory) };
        let _ = memory;
        // mem::forget(memory);

        return Ok(val);
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

        let mut line = 1;
        let mut column = 1;

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
            else if c.is_whitespace() {
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
                else if !buf.trim().is_empty() {
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
        let input = 
// entry:
"
    add rout 12
    add r0 3
    add r0 3
    add r1 23424324
    add r2 444
    add rin 2
    add rip 2
"
    // add [rout] 3
    // add rin [rout]
    // ret
;//read_line().expect("Realistically a really bad error.");

        println!("Register at {reg}: {byte}", byte=machine.get_reg_value(reg).expect("Tried to read unknown register"));

        let Ok(bytecode) = machine.assemble(input) else {
            panic!("beuh");
        };

        println!("Running the following code: \n{bytecode:?}");

        machine.run_bytecode(bytecode.as_slice());
        
        let read_loc = 12;
        println!("Byte at {read_loc:#x}: {byte}", byte=machine.read_byte(read_loc).expect("Out of bounds memory access."));
        machine.print_registers();
        // println!("Register at {reg}: {byte}", byte=machine.get_reg_value(reg).expect("Tried to read unknown register"));
    // }
}
