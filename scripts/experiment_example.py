from declutr_experiment import DeclutrExperiment

# Simple to run Declutr encoding model experiment.
experiment = DeclutrExperiment(variable_arg="use_positional_encodings",
                               variable_domain=[True, False],
                               constant_arg_vals=dict(encoder_model="transformer_encoder", sampling=.0005),
                               add_constants_to_id=["encoder_model"])
experiment.run()