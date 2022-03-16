use ::rand::prelude::StdRng;

use auto_diff::Var;


pub fn normal(data: &Var, mean: Option<Var>, std: Option<Var>, rng: &mut StdRng) {
    let size = data.size();
    data.set(&Var::normal(&mut rng, &size, 0., 1.)); // TODO use args.
}
