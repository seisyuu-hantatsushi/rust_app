use std::{env,mem};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, Read, SeekFrom};

use binary_layout::prelude::*;
use clap::{App,Arg};

use eframe::egui;
use egui_extras::RetainedImage;

define_layout!(minst_label_header, BigEndian, {
    magic: u32,
    num_of_images: u32,
});

define_layout!(minst_image_header, BigEndian, {
    magic: u32,
    num_of_images: u32,
    row: u32,
    col: u32
});

#[derive(Debug,Clone)]
struct AppContext {
    labels_file: String,
    images_file: String,
    no_of_image: u32,
    row_of_image: u32,
    col_of_image: u32,
    label_of_image: u8,
    image_data: Option<Box<[u8]>>,
    is_need_revered: bool
}

struct GuiContext {
    appCtx : AppContext
}

impl eframe::App for GuiContext {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
	let label_table:[char;47] = ['0','1','2','3','4','5','6','7','8','9',
				     'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
				     'a','b',    'd','e','f','g','h',                   'n',         'q','r','t'];
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Show MINST image");
	    ui.horizontal(|ui| {
		ui.label("labels file:");
		ui.label(self.appCtx.labels_file.to_owned());
	    });
	    ui.horizontal(|ui| {
		ui.label("images_file:");
		ui.label(self.appCtx.images_file.to_owned());
	    });
	    ui.horizontal(|ui| {
		ui.label("select image no:");
		ui.label(self.appCtx.no_of_image.to_string().to_owned());
	    });
	    ui.horizontal(|ui| {
		ui.label("label of image:");
		ui.label(self.appCtx.label_of_image.to_string().to_owned());
		ui.label("(");
		ui.label(label_table[self.appCtx.label_of_image as usize].to_string());
		ui.label(")");
	    });

	    let to_color_image = |image:&Box<[u8]>| -> egui::ColorImage {
		let data_vec = vec![0; (self.appCtx.row_of_image*self.appCtx.col_of_image*4) as usize];
		let mut data_boxed:Box<[u8]> = data_vec.into_boxed_slice();
		for c in 0..self.appCtx.col_of_image {
		    for r in 0..self.appCtx.row_of_image {
			let dst_pos = (c*self.appCtx.row_of_image*4+r*4) as usize;
			let src_pos = (c*self.appCtx.row_of_image+r) as usize;
			data_boxed[dst_pos+0] = image[src_pos];
			data_boxed[dst_pos+1] = image[src_pos];
			data_boxed[dst_pos+2] = image[src_pos];
			data_boxed[dst_pos+3] = 255;
		    }
		}
		egui::ColorImage::from_rgba_unmultiplied([self.appCtx.row_of_image as usize ,self.appCtx.col_of_image as usize],&data_boxed)
	    };

	    if let Some(ref image) = self.appCtx.image_data {
		let minst_image = RetainedImage::from_color_image("minst_image", to_color_image(&image));
		minst_image.show(ui);
		//ui.add(egui::Image::new(minst_image.texture_id(ctx),
		//minst_image.size_vec2()).rotate(90.0_f32.to_radians(), egui::Vec2::splat(0.5)));
	    }
	});
    }
}

fn main() {

    let appArgs = App::new("minst_reader")
	.version("0.1.0")
	.arg(Arg::with_name("labels_file")
	     .help("minst label file")
	     .short('l')
	     .long("labels_file")
	     .takes_value(true)
	     .required(true))
	.arg(Arg::with_name("images_file")
	     .help("minst image file")
	     .short('i')
	     .long("images_file")
	     .takes_value(true)
	     .required(true))
	.arg(Arg::with_name("no_of_image")
	     .help("specified no of image")
	     .short('n')
	     .long("no_of_image")
	     .default_value("-1")
	     .takes_value(true))
	.arg(Arg::with_name("emnist")
	     .help("specified emnist format")
	     .long("emnist")
	     .takes_value(false));

    let mut ctx:AppContext = match appArgs.try_get_matches() {
	Ok(m) => {
	    println!("{:?}",m);
	    let no_of_image = m.value_of("no_of_image").unwrap().parse::<i32>().unwrap();
	    let ctx = AppContext {
		labels_file: String::from(m.value_of("labels_file").unwrap()),
		images_file: String::from(m.value_of("images_file").unwrap()),
		no_of_image: if no_of_image == -1 { 0 } else { no_of_image as u32 },
		row_of_image: 0,
		col_of_image: 0,
		label_of_image: 0,
		image_data: None,
		is_need_revered: m.contains_id("emnist")
	    };
	    println!("{:?}",ctx.is_need_revered);
	    ctx
	},
	Err(e) => {
	    println!("Error {}", e);
	    return;
	}
    };

    println!("labels {}", ctx.labels_file);
    println!("images {}", ctx.images_file);

    let mut labels_file = BufReader::new(
	File::open(&ctx.labels_file).expect("Failed to open file")
    );

    let mut images_file = BufReader::new(
	File::open(&ctx.images_file).expect("Failed to open file")
    );

    {
	let mut header:[u8;8] = unsafe{ mem::MaybeUninit::uninit().assume_init() };

	let _readsize = labels_file.read(&mut header).unwrap_or_else(|err| panic!("IO Error => {}", err));
	let minst_label_header = minst_label_header::View::new(header);
	println!("magic {:08x}", minst_label_header.magic().read());
	println!("num of images {}", minst_label_header.num_of_images().read());
    }

    {
	let mut header:[u8;16] = unsafe{ mem::MaybeUninit::uninit().assume_init() };

	let _readsize = images_file.read(&mut header).unwrap_or_else(|err| panic!("IO Error => {}", err));
	let minst_images_header = minst_image_header::View::new(header);
	println!("magic {:08x}", minst_images_header.magic().read());
	println!("num of images {}", minst_images_header.num_of_images().read());
	println!("num of row {}", minst_images_header.row().read());
	println!("num of col {}", minst_images_header.col().read());

	ctx.row_of_image = minst_images_header.row().read();
	ctx.col_of_image = minst_images_header.col().read();
    }

    println!("no of image: {}", ctx.no_of_image);

    {
	let mut data:[u8;1] = unsafe{ mem::MaybeUninit::uninit().assume_init() };
	labels_file.seek(SeekFrom::Current(ctx.no_of_image as i64));
	labels_file.read(&mut data).unwrap_or_else(|err| panic!("IO Error => {}", err));
	ctx.label_of_image = data[0];
    }

    {
	let start_pos = (ctx.row_of_image*ctx.col_of_image*ctx.no_of_image) as i64;
	let data = vec![0; (ctx.row_of_image*ctx.col_of_image) as usize];
	let mut data_boxed:Box<[u8]> = data.into_boxed_slice();
	images_file.seek(SeekFrom::Current(start_pos as i64));
	images_file.read(&mut data_boxed);
	ctx.image_data = Some(data_boxed);
    }

    ctx.image_data = match ctx.image_data {
	Some(mut image) => {
	    if ctx. is_need_revered {
		let data = vec![0; (ctx.row_of_image*ctx.col_of_image) as usize];
		let mut reversed_image:Box<[u8]> = data.into_boxed_slice();
		/*
		for c in 0..ctx.col_of_image as usize {
		for r in 0..ctx.row_of_image as usize {
		let pos = (c*(ctx.row_of_image as usize)+r) as usize;
		print!("{:02x}", image[pos]);
	    }
		println!("");
	    }
		println!("");
		 */

		// 右90度回転
		for c in 0..ctx.col_of_image as usize {
		    for r in 0..ctx.row_of_image as usize {
			let src_pos = ((ctx.col_of_image-1) as usize - r)*(ctx.row_of_image as usize)+c;
			reversed_image[c*(ctx.row_of_image as usize)+r] = image[src_pos];
		    }
		}
		/*
		for c in 0..ctx.col_of_image as usize {
		for r in 0..ctx.row_of_image as usize {
		let pos = (c*(ctx.row_of_image as usize)+r) as usize;
		print!("{:02x}", reversed_image[pos]);
	    }
		println!("");
	    }
		println!("");
		 */
		// 鏡像反転
		for c in 0..ctx.col_of_image as usize {
		    for r in 0..ctx.row_of_image as usize {
			let src_pos = c*(ctx.row_of_image as usize) + ((ctx.row_of_image-1) as usize - r);
			image[c*(ctx.row_of_image as usize)+r] = reversed_image[src_pos];
		    }
		}
	    }
	    Some(image)
	},
	None => None
    };

    if ctx.image_data.is_none() {
	return
    }

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(GuiContext { appCtx: ctx })),
    );
}
