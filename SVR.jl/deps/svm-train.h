extern struct svm_parameter param;		// set by parse_command_line
extern struct svm_problem prob;		// set by read_problem
extern struct svm_model *model;
extern struct svm_node *x_space;
extern int cross_validation;
extern int nr_fold;
void read_problem(const char *filename);
void *get_prob();
void *get_param();