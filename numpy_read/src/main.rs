
use std::{env,mem};
use std::fs::File;
use std::path::Path;
use std::io::prelude::*;
use std::io::{BufReader, Read};
use std::collections::HashMap;
use byteorder::{ByteOrder, LittleEndian};

use clap::{App,Arg};
use hdf5;

#[allow(dead_code)]
fn get_hex_rep(byte_array: &[u8]) -> String {
    let build_string_vec: Vec<String> = byte_array.iter().enumerate()
        .map(|(i, val)| {
            if i == 7 { format!("{:02x} ", val) }
            else { format!("{:02x}", val) }
        }).collect();
    build_string_vec.join(" ")
}

#[derive(Debug)]
enum Endian {
    Little, Big
}

#[derive(Debug)]
enum ValueType {
    UInt8, Int8,
    UInt16, Int16,
    UInt32, Int32,
    UInt64, Int64,
    Float32, Float64
}

#[derive(Debug)]
struct NpyFormat {
    pub value_type : ValueType,
    pub endian : Endian,
    pub fortran_order: bool,
    pub shape: (usize, usize)
}

#[derive(Debug)]
struct NpyArray {
    pub shape: (usize, usize),
    pub values: Box<[f64]>
}

fn parse_npy_format(format_str: String) -> Result<NpyFormat,(u32, String)> {

    #[derive(PartialEq)]
    enum ParseState {
	SearchLeftCurlyBracket,
	SearchKeyQuato,
	StoreKey,
	SearchSpliter,
	SearchValueStr,
	StoreValue,
	StoreValueParentheses,
	SearchComma
    }

    let mut dict:HashMap<String, String> = HashMap::new();
    {
	let s = format_str.trim().chars();
	let mut state: ParseState = ParseState::SearchLeftCurlyBracket;
	let mut key:Vec<char> = Vec::new();
	let mut value:Vec<char> = Vec::new();

	//println!("{}", format_str);

	for c in s {
	    match c {
		'{' => {
		    if state == ParseState::SearchLeftCurlyBracket {
			state = ParseState::SearchKeyQuato;
		    }
		    else if state == ParseState::StoreValue {
			value.push(c);
		    }
		    else if state == ParseState::StoreKey {
			key.push(c);
		    }
		    else {
			return Err((4,"Invalid format string".to_string()));
		    }
		},
		'}' => {
		    if state ==  ParseState::SearchComma {
			break;
		    }
		},
		'\'' => {
		    if state == ParseState::SearchKeyQuato {
			key.clear();
			state = ParseState::StoreKey;
		    }
		    else if state == ParseState::StoreKey {
			state = ParseState::SearchSpliter;
		    }
		    else if state == ParseState::SearchValueStr {
			value.push(c);
			state = ParseState::StoreValue;
		    }
		    else if state == ParseState::StoreValue {
			value.push(c);
		    }
		    else {
			return Err((4,"Invalid format string".to_string()));
		    }
		},
		':' => {
		    if state == ParseState::SearchSpliter {
			state = ParseState::SearchValueStr;
		    }
		    else if state == ParseState::StoreValue {
			value.push(c);
		    }
		    else if state == ParseState::StoreKey {
			key.push(c);
		    }
		}
		' ' => {
		    if state == ParseState::StoreValue {
			value.push(c);
		    }
		    else if state == ParseState::StoreKey {
			key.push(c);
		    }
		},
		',' => {
		    if state == ParseState::StoreValue {
			dict.insert(key.iter().collect::<String>(), value.iter().collect::<String>());
			state = ParseState::SearchKeyQuato;
		    }
		    else if state == ParseState::StoreKey {
			key.push(c);
		    }
		    else if state == ParseState::StoreValueParentheses {
			value.push(c);
		    }
		}
		'(' => {
		    if state == ParseState::StoreKey {
			key.push(c);
		    }
		    else if state == ParseState::SearchValueStr {
			value.clear();
			value.push(c);
			state = ParseState::StoreValueParentheses;
		    }
		},
		')' => {
		    if state == ParseState::StoreKey {
			key.push(c);
		    }
		    else if state == ParseState::StoreValueParentheses {
			value.push(c);
			state = ParseState::StoreValue;
		    }
		},
		_ => {
		    if state == ParseState::SearchValueStr {
			value.clear();
			value.push(c);
			state = ParseState::StoreValue;
		    }
		    else if state == ParseState::StoreKey {
			key.push(c);
		    }
		    else if state == ParseState::StoreValue {
			value.push(c);
		    }
		    else if state == ParseState::StoreValueParentheses {
			value.push(c);
		    }
		}
	    }
	}
    }

    let (endian, value_type) = if let Some(descr) = dict.get("descr") {
	let descr_str = descr.trim_matches('\'');
	(if descr_str.chars().nth(0) == Some('<') {
	    Endian::Little
	}
	 else {
	     Endian::Big
	 },
	 if &descr_str[1..] == "f4" {
	     ValueType::Float32
	 }
	 else if &descr_str[1..] == "f8" {
	     ValueType::Float64
	 }
	 else if &descr_str[1..] == "u8" {
	     ValueType::UInt8
	 }
	 else {
	     ValueType::UInt8
	 })
    }
    else {
	return Err((5,"not found descr key".to_string()));
    };

    let fortran_order = if let Some(order) = dict.get("fortran_order") {
	if order == "True" {
	    true
	}
	else if order == "False" {
	    false
	}
	else {
	    return Err((5,"not found fortran_order key".to_string()));
	}
    }
    else {
	return Err((5,"not found fortran order key".to_string()));
    };

    let shape = if let Some(shape_str) = dict.get("shape") {
	let shape_token:Vec<&str> = shape_str.
	    trim_start_matches('(').trim_end_matches(')').trim_end_matches(',').split(',').collect::<Vec<&str>>();

	if shape_token.len() == 1 {
	    match shape_token[0].parse() {
		Ok(r) => (1, r),
		Err(_) => {
		    return Err((5,"invalid shape value".to_string()));
		}
	    }
	}
	else if shape_token.len() == 2 {
	    match (shape_token[0].parse(), shape_token[1].parse()) {
		(Ok(c), Ok(r)) => {
		    (c, r)
		}
		_ => {
		    return Err((5,"invalid shape value".to_string()));
		}
	    }
	}
	else {
	    return Err((5,"invalid shape value".to_string()));
	}
    }
    else {
	return Err((5,"not found shape key".to_string()));
    };

    Ok(NpyFormat {
	value_type: value_type,
	endian: endian,
	fortran_order: fortran_order,
	shape: shape
    })
}

fn read_magic(mut f:BufReader<File>) -> Result<NpyArray,(u32,String)>{

    match f.fill_buf() {
	Ok(b) => {
	    if b.len() == 0 {
		return Err((3, "file size is too short".to_string()));
	    }
	},
	Err(err) => {
	    return Err((2, err.to_string()));
	}
    }

    let mut magic:[u8;6] = unsafe{ mem::MaybeUninit::uninit().assume_init() };
    match f.read(&mut magic) {
	Ok(readsize) => {
	    if b"\x93NUMPY" != &magic {
		return Err((1,"magic is invalid".to_string()));
	    }
	},
	Err(err) =>{
	    println!("Error {}", err);
	    return Err((2, err.to_string()));
	}
    }

    read_header(f)

}

fn read_header(mut f:BufReader<File>) -> Result<NpyArray,(u32,String)> {
    let mut version = [0u8,2];
    let mut padding_size:usize = 0;

    match f.fill_buf() {
	Ok(b) => {
	    if b.len() == 0 {
		return Err((3, "file size is too short".to_string()));
	    }
	},
	Err(err) => {
	    return Err((2, err.to_string()));
	}
    }

    match f.read(&mut version) {
	Ok(readsize) => {
	    if version == [1,0] {
		let mut padding_length = [0u8,2];
		match f.read(&mut padding_length) {
		    Ok(readsize) => {
			padding_size = LittleEndian::read_u16(&padding_length) as usize;
		    },
		    Err(err) => {
			println!("Error {}", err);
			return Err((2, err.to_string()));
		    }
		}
	    }
	    else if version == [2,0] {
		let mut padding_length = [0u8,4];
		match f.read(&mut padding_length) {
		    Ok(readsize) => {
			padding_size = LittleEndian::read_u32(&padding_length) as usize;
		    },
		    Err(err) => {
			println!("Error {}", err);
			return Err((2, err.to_string()));
		    }
		}
	    }
	    else {
		return Err((3,"Unknown format version".to_string()));
	    }
	},
	Err(err) => {
	    println!("Error {}", err);
	    return Err((2, err.to_string()));
	}
    }

    read_format(f, padding_size)

}

fn read_format(mut f:BufReader<File>, padding_size:usize) -> Result<NpyArray,(u32,String)> {

    match f.fill_buf() {
	Ok(b) => {
	    if b.len() == 0 {
		return Err((3, "file size is too short".to_string()));
	    }
	},
	Err(err) => {
	    return Err((2, err.to_string()));
	}
    };

    let buffer = f.buffer();
    let npy_format = match parse_npy_format(String::from_utf8(buffer[0..padding_size].to_vec()).unwrap()){
	Ok(format) => {
	    format
	}
	Err(err) => {
	    println!("Err {:?}", err);
	    return Err(err);
	}
    };
    f.consume(padding_size);

    read_value(f, npy_format)

}

fn read_value(mut f:BufReader<File>, format: NpyFormat) -> Result<NpyArray,(u32,String)> {

    let (c,r) = format.shape;
    let mut values:Vec<f64> = Vec::with_capacity(c*r);
    let mut read_counter = 0;
    let (mut i, mut j) = (0,0);

    unsafe { values.set_len(c*r) };


    loop {
	match f.fill_buf() {
	    Ok(b) => {
		if b.len() == 0 {
		    if read_counter == c*r {
			return Ok(NpyArray {
			    shape: (c, r),
			    values: values.into_boxed_slice()
			});
		    }
		    else {
			return Err((3, "file size is too short".to_string()));
		    }
		}
	    },
	    Err(err) => {
		return Err((2, err.to_string()));
	    }
	};

	values[i*r+j] = match format.value_type {
	    ValueType::Float32 => {
		let mut value_raw = [0u8;4];
		match f.read(&mut value_raw) {
		    Ok(readsize) => {
			match format.endian {
			    Endian::Little => {
				f32::from_le_bytes(value_raw) as f64
			    }
			    _ => {
				f32::from_be_bytes(value_raw) as f64
			    }
			}
		    },
		    Err(err) => {
			return Err((2, err.to_string()));
		    }
		}
	    },
	    _ => {
		return Err((8,"not support value format ".to_string()));
	    }
	};

	if format.fortran_order {
	    // Fortran order
	    i += 1;
	    if i >= format.shape.0 {
		i = 0;
		j += 1;
	    }
	}
	else {
	    // normal order
	    j += 1;
	    if j >= format.shape.1 {
		j  = 0;
		i += 1;
	    }
	};

	read_counter += 1;
    }

}

fn parse_npy(file:&str) ->  Result<NpyArray,(u32,String)> {
    let mut f = BufReader::new(
	File::open(&file).expect(format!("Failed to open file. {}", file).as_str())
    );

    read_magic(f)
}

fn main() {

    struct AppContext {
	weight_files : Vec<String>,
	output_file : Option<String>
    }

    let app_args = App::new("npy_reader")
	.version("0.1.0")
	.arg(Arg::new("npy_files").
	     multiple(true))
	.arg(Arg::new("output_file")
	     .short('o')
	     .long("output_file")
	     .takes_value(true));

    let ctx = match app_args.try_get_matches(){
	Ok(m) => {
	    let files:Vec<&str> = m.values_of("npy_files").unwrap().collect();
	    AppContext {
		weight_files : files.into_iter().map(String::from).collect(),
		output_file : match m.value_of("output_file") {
		    Some(str) => Some(str.to_string()),
		    None => None
		}
	    }
	},
	Err(e) => {
	    println!("Error {}", e);
	    return;
	}
    };

    let output:Option<hdf5::File> = if let Some(output_file) = ctx.output_file {
	if let Ok(h) = hdf5::File::create(output_file) {
	    Some(h)
	}
	else {
	    None
	}
    }
    else {
	None
    };

    for weight_file in ctx.weight_files {
	match parse_npy(&weight_file) {
	    Ok(weight_array) => {
		/*
		println!("{} {}", weight_array.shape.0, weight_array.shape.1);
		println!("{:?}", &weight_array.values[weight_array.shape.1*0+0..weight_array.shape.1*1]);
		println!("{}", weight_array.values[weight_array.shape.1*0+1]);
		println!("{}", weight_array.values[weight_array.shape.1*1+0]);
		println!("{}", weight_array.values[weight_array.shape.1*10+12]);
		println!("{:?}", &weight_array.values[weight_array.shape.1*783..weight_array.shape.1*784]);
*/
		if let Some(ref output) = output {
		    let file_path = Path::new(&weight_file);
		    let array = &weight_array.values;
		    let wrapped_date_set = output.new_dataset::<f32>().shape([weight_array.shape.0, weight_array.shape.1]).create(file_path.file_stem().unwrap().to_str());
		    if let Ok(data_set) = wrapped_date_set {
			if let Err(err) = data_set.write_raw(array) {
			    match err {
				hdf5::Error::HDF5(es) => {
				    if let Ok(expanded_es) = es.expand() {
					eprintln!("{}",expanded_es.description().to_string());
				    }
				    else {
					eprintln!("Internal Error");
				    }
				},
				hdf5::Error::Internal(es) => {
				    eprintln!("{}",es.to_string());
				}
			    }
			}
		    }
		    else {
			eprintln!("failed to write hdf5 file");
			return;
		    }
		}
	    },
	    Err((code, message)) => {
		println!("{} {}", code, message);
		return;
	    }
	}
    }
}
