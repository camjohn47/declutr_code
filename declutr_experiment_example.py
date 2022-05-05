from declutr_experiment import DeclutrExperiment

# Simple to run Declutr encoding model experiment.
experiment = DeclutrExperiment(variable_arg="encoding_model", variable_domain=["rnn", "transformer"])
experiment.run()