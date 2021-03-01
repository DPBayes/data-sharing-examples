from collections import OrderedDict

independent_model = OrderedDict()
independent_model['age'] = 'Beta'
independent_model['per'] = 'Beta'
independent_model['C10AA.DDD'] = 'Bernoulli'
independent_model['DM.type'] = 'Categorical'
independent_model['shp'] = 'Categorical'
independent_model['lex.dur'] = 'Beta'
independent_model['G03.DDD'] = 'Bernoulli'
independent_model['.i.cancer'] = 'Bernoulli'
independent_model['ep'] = 'Bernoulli'
independent_model['dead'] = 'Bernoulli'
independent_model['pi_unconstrained'] = 'Categorical'

dependent_model = OrderedDict()
dependent_model['age'] = 'Beta'
dependent_model['per'] = 'Beta'
dependent_model['C10AA.DDD'] = 'Bernoulli'
dependent_model['DM.type'] = 'Categorical'
dependent_model['shp'] = 'Categorical'
dependent_model['lex.dur'] = None
dependent_model['G03.DDD'] = 'Bernoulli'
dependent_model['.i.cancer'] = 'Bernoulli'
dependent_model['ep'] = None 
dependent_model['dead'] = None
dependent_model['pi_unconstrained'] = 'Categorical'

cat_ade_model = OrderedDict()
cat_ade_model['age'] = 'Beta'
cat_ade_model['per'] = 'Beta'
cat_ade_model['C10AA.DDD'] = 'Bernoulli'
cat_ade_model['DM.type'] = 'Categorical'
cat_ade_model['shp'] = 'Categorical'
cat_ade_model['lex.dur'] = None
cat_ade_model['G03.DDD'] = 'Bernoulli'
cat_ade_model['.i.cancer'] = 'Bernoulli'
cat_ade_model['ade'] = 'Categorical'
cat_ade_model['pi_unconstrained'] = 'Categorical'
