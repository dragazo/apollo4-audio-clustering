#ifndef A3EM_APP_AI_H
#define A3EM_APP_AI_H

/// Given a 1-D tensor `input` (with len `input_len`) at a given `sample rate`, generates the preprocessed output for inference.
/// `output`, `output_rows`, `output_cols`, and `output_deleter` are pointers to storage locations for the result, but otherwise do not need to be initialized.
///
/// The `output` value is dynamically allocated and will need to be passed to the resulting `output_deleter` manually.
void preprocess(float *input, unsigned input_len, unsigned sample_rate, float **output, unsigned *output_rows, unsigned *output_cols, void (**output_deleter)(float*));

#endif
