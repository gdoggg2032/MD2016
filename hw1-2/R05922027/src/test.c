#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define MAX_STATE 55
#define MAX_SEQ 2001115




typedef struct{

	int state_num; 
	double transition[MAX_STATE][MAX_STATE];
	double emission[MAX_STATE][MAX_STATE];
	double initial[MAX_STATE];

}Model;

Model model;
FILE* fp;
FILE* fp_result;
int observ[MAX_SEQ];
int observ_num = 2000114;

double Delta[MAX_SEQ][MAX_STATE];
int traceback[MAX_SEQ][MAX_STATE];





void loadModel(Model* m, char* initialFile, char* transitionFile, char* emissionFile) {

	int i, j;
	double b;

	m->state_num = 0;

	fp = fopen(initialFile, "r");
	i = 0;
	while(fscanf(fp, "%lf\n", &b) > 0) {
		m->initial[i] = b;
		i++;
	}

	m->state_num = i;


	fp = fopen(transitionFile, "r");
	while(fscanf(fp, "%d %d %lf\n", &i, &j, &b) > 0) {
		m->transition[i][j] = b;
	}

	fp = fopen(emissionFile, "r");
	while(fscanf(fp, "%d %d %lf\n", &i, &j, &b) > 0) {
		m->emission[i][j] = b;
	}

}



void viterbi(int* ans) {
	int i, j, n, t;
	int N, T;

	N = model.state_num;
	T = observ_num;

	for(i = 0; i < N; i++){
		Delta[0][i] = model.initial[i] + model.emission[i][observ[0]];
		traceback[0][i] = 0;
	}
		

	for(t = 1; t < T; t++) {
		for(i = 0; i < N; i++) {

			double best_delta = -INFINITY;
			int best_n;
			for(n = 0; n < N; n++) {
				double tmp = Delta[t-1][n] + model.transition[n][i];
				if(tmp > best_delta) {
					best_delta = tmp;
					traceback[t][i] = n;
				}
			}
			Delta[t][i] = best_delta + model.emission[i][observ[t]];
		}
	}

	int max_i;
	double max_prob = -INFINITY;
	for(i = 0; i < N; i++) {
		if(Delta[T-1][i] > max_prob) {
			max_i = i;
			max_prob = Delta[T-1][i];
		}
	}

	//traceback
	for(t = T-1; t >= 0; t--) {
		ans[t] = max_i;
		max_i = traceback[t][max_i];
	}

	return;

}

int get_observ() {
	int i = 0;
	int o;

	observ[i] = model.state_num - 1;
	i += 1;
	while(fscanf(fp, "%d ", &o) > 0) {
		observ[i] = o;
		i += 1;
		if(o == model.state_num - 1) { // o == ' '
			break;
		}
	}
	observ_num = i;
	return observ_num;
}


int main(int argc, char * argv[]) {


	int i;

	char* initialFile = argv[1];
	char* transitionFile = argv[2];
	char* emissionFile = argv[3];

	char* inputFile = argv[4];
	char* outputFile = argv[5];

	fprintf(stderr, "load model\n");
	loadModel(&model, initialFile, transitionFile, emissionFile);



	fprintf(stderr, "starting testing\n");

	fp = fopen(inputFile, "r");
	fp_result = fopen(outputFile, "w");

	while(get_observ() > 1) {

		int ans[observ_num];
		viterbi(ans);
		for(i = 1; i < observ_num; i++)
			fprintf(fp_result, "%d ", ans[i]);
	}

	fprintf(stderr,"testing completed\n");





	return 0;
}