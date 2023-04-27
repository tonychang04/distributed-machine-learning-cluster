use tch::nn::{FuncT, ModuleT};
use tch::TchError;
use tch::vision::imagenet;


// simple demo of how to use the imagenet module
pub fn sample_predict() -> Result<(), TchError> {
    let image = imagenet::load_image_and_resize("test_files/imagenet_1k/train/n01491361/ILSVRC2012_00034413.JPEG", 224, 224)?;

    // A variable store is created to hold the model parameters.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    // let resnet18 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    let alexnet = tch::vision::alexnet::alexnet(&vs.root(), imagenet::CLASS_COUNT);

    vs.load("pretrained_models/alexnet.ot")?;

    let output = alexnet.forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float);
    let top = imagenet::top(&output, 1);
    print!("Top 1: {:?}", top[0].1); // prints out the string of corresponding label
    Ok(())
}

pub fn load_model(model_name: String) -> Box<dyn ModuleT> {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    let model: Box<dyn ModuleT> = match model_name.as_str() {
        "resnet18" => Box::new(tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT)),
        "alexnet" => Box::new(tch::vision::alexnet::alexnet(&vs.root(), imagenet::CLASS_COUNT)),
        _ => panic!("Model not found"),
    };

    vs.load(format!("{}.ot", model_name)).unwrap_or(());

    return model;
}

pub fn predict_image(model: &Box<dyn ModuleT>, image_path: String, label: String) -> Option<bool> {
    let image = imagenet::load_image_and_resize(image_path.as_str(), 224, 224).ok()?;

    let output = model.forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float);
    let prediction = imagenet::top(&output, 1);

    return if prediction[0].1 == label {
        Some(true)
    } else {
        Some(false)
    }
}




