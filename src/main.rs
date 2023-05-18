const INPUT_LEN: usize = 4;
const CLASSES: usize = 3;
const DATA_PATH: &'static str = "iris_data_files/iris_training.dat";
const TEST_PATH: &'static str = "iris_data_files/iris_test.dat";

const fn class_name(n: usize) -> &'static str {
    match n {
        1 => "setosa",
        2 => "versicolor",
        3 => "virginica",
        _ => unreachable!(),
    }
}

#[derive(Debug, Clone, Copy)]
struct TrainingData {
    input: [f64; INPUT_LEN],
    class: usize,
}

#[derive(Debug)]
struct ClassData {
    stdev: Vec<f64>,
    meandev: Vec<f64>,
    prob: f64,
}

impl ClassData {
    fn new() -> Self {
        Self {
            stdev: vec![],
            meandev: vec![],
            prob: 0.,
        }
    }
}

struct Classifier {
    classes: Vec<ClassData>,
}

impl Classifier {
    fn new() -> Self {
        Self { classes: vec![] }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).fold(0., |acc, (a, b)| acc + a * b)
    }

    fn mean(data: &[f64]) -> f64 {
        data.iter().map(|e| *e).sum::<f64>() / (data.len() as f64)
    }

    fn stdev(data: &[f64]) -> f64 {
        let variance = (Self::dot(data, data) / data.len() as f64) - Self::mean(data).powi(2);
        variance.sqrt()
    }

    fn prob_of(val: f64, mean: f64, stdev: f64) -> f64 {
        const INV_SQRT_TAU: f64 = 0.3989422804014327;
        let z = (val - mean) / stdev;
        let inv_sqrt = INV_SQRT_TAU / stdev;

        let prob = inv_sqrt * (z * z * -0.5).exp();

        prob
    }

    fn calc_class_data(data: &[TrainingData], class: usize) -> ClassData {
        let mut cd = ClassData::new();
        let class_data = data
            .iter()
            .filter(|TrainingData { input: _, class: c }| *c == class)
            .collect::<Vec<_>>();

        for TrainingData {
            input: row,
            class: _,
        } in &class_data
        {
            cd.meandev.push(Self::mean(row));
            cd.stdev.push(Self::stdev(row));
        }

        cd.prob = class_data.len() as f64 / data.len() as f64;
        cd
    }

    fn fit(&mut self, data: &[TrainingData]) {
        for class in 0..CLASSES {
            self.classes.push(Self::calc_class_data(&data, class + 1));
        }
    }

    fn class_prob(test_data: [f64; INPUT_LEN], cd: &ClassData) -> f64 {
        let mut p = 1.;

        for i in 0..INPUT_LEN {
            let mean = cd.meandev[i + cd.meandev.len() / 5];
            let st = cd.stdev[i + (cd.meandev.len() / 5)];
            p *= Self::prob_of(test_data[i], mean, st);
        }

        p *= cd.prob;
        p
    }

    fn predict(&self, data: [f64; INPUT_LEN]) -> usize {
        let mut res = vec![];
        for class in 0..CLASSES {
            res.push(Self::class_prob(data, &self.classes[class]));
        }
        res.iter()
            .position(|x| x == res.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .unwrap()
            + 1
    }
}

fn load_dataset() -> (Vec<TrainingData>, Vec<TrainingData>) {
    let contents = std::fs::read_to_string(DATA_PATH).expect("Invalid training data path");
    let mut data1 = vec![];
    for line in contents.lines()  {
        let split = line.split_whitespace().collect::<Vec<_>>();
        let mut input = [0.; INPUT_LEN];
        for i in 0..INPUT_LEN {
            input[i] = split[i].parse().expect("Invalid number in training data");
        }
        let class: usize = split[INPUT_LEN].parse().unwrap();

        data1.push(TrainingData { input, class });
    }

    let contents = std::fs::read_to_string(TEST_PATH).expect("Invalid training data path");
    let mut data2 = vec![];
    for line in contents.lines() {
        let split = line.split_whitespace().collect::<Vec<_>>();
        let mut input = [0.; INPUT_LEN];
        for i in 0..INPUT_LEN {
            input[i] = split[i].parse().expect("Invalid number in training data");
        }
        let class: usize = split[INPUT_LEN].parse().unwrap();

        data2.push(TrainingData { input, class });
    }
    (data1, data2)
}

fn main() {
    let (training, test) = load_dataset();
    let mut model = Classifier::new();
    model.fit(&training);

    let mut right = 0.;
    let mut wrong = 0.;

    for row in test {
        let predicted = model.predict(row.input);
        println!("{}, {}", class_name(row.class), class_name(predicted));
        if predicted == row.class {
            right += 1.;
        } else {
            wrong += 1.;
        }
    }

    println!("model accuracy: {}", right / (right + wrong));

    let setosa = [5.1, 3.5, 1.4, 0.2];
    let versicolor = [5.5, 2.4, 3.8, 1.1];
    let virginica = [6.7, 3.1, 5.6, 2.4];

    println!(
        "should be setosa: {}", 
        class_name(model.predict(setosa))
    );
    println!(
        "should be versicolor: {}",
        class_name(model.predict(versicolor))
    );
    println!(
        "should be virginica: {}",
        class_name(model.predict(virginica))
    );
}

