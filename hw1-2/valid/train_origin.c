#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#define MAX_STATE 55
#define MAX_SEQ 2000115


typedef struct{

	int state_num; 
	double transition[MAX_STATE][MAX_STATE];
	double emission[MAX_STATE][MAX_STATE];
	double initial[MAX_STATE];

}Model;

Model model;

double Alpha[MAX_SEQ][MAX_STATE];
double Beta[MAX_SEQ][MAX_STATE];
double Gamma[MAX_SEQ][MAX_STATE];
double Epsilon[MAX_SEQ][MAX_STATE][MAX_STATE];

double acc_gamma[MAX_STATE];
double acc_gamma_T[MAX_STATE];
double acc_gamma_observ[MAX_STATE][MAX_STATE];
double acc_epsilon[MAX_STATE][MAX_STATE];
double new_initial[MAX_STATE];

int observ[MAX_SEQ];
int observ_num = 2000114;

FILE * fp;

int emission_mask[MAX_STATE][MAX_STATE];

void loadModel(Model* m, char *emissionFile) {
	// load emission probs
	double emission_sum[MAX_STATE];
	int i, j;
	double b;
	fp = fopen(emissionFile, "r");

	m->state_num = 0;

	while(fscanf(fp, "%d %d %lf\n", &i, &j, &b) > 0) {
		if(b > 0) {
			m->emission[i][j] = b;
			emission_mask[i][j] = 1;
		}
		else {
			m->emission[i][j] = 0;
			emission_mask[i][j] = 0;
		}
		emission_sum[i] += b;
		if(m->state_num < i+1)
			m->state_num = i+1;
		if(m->state_num < j+1)
			m->state_num = j+1;
	}

	// normalize
	for(i = 0; i < m->state_num; i++) {
		for(j = 0; j < m->state_num; j++) {
			m->emission[i][j] /= emission_sum[i];
		}
	}

	// uniformly assign transition probs and initial probs

	for(i = 0; i < m->state_num; i++) {
		m->initial[i] = 1.0 / m->state_num;
		for(j = 0; j < m->state_num; j++) {
			m->transition[i][j] = 1.0 / m->state_num;
		}
	}

	return;
		
}

void dumpModel(Model* m, char* dumpInitialFile, char* dumpTransitionFile, char* dumpEmissionFile) {
	
	int i, n;
	int N = model.state_num;
	
	//initial
	fp = fopen(dumpInitialFile, "w");
	for(i = 0; i < N; i++) {
		fprintf(fp, "%lf\n", m->initial[i]);
	}

	//transition
	fp = fopen(dumpTransitionFile, "w");
	for(i = 0; i < N; i++) {
		for(n = 0; n < N; n++) {
			fprintf(fp, "%d %d %lf\n", i, n, m->transition[i][n]);
		}
	}

	//emission
	fp = fopen(dumpEmissionFile, "w");
	for(i = 0; i < N; i++) {
		for(n = 0; n < N; n++) {
			fprintf(fp, "%d %d %lf\n", i, n, m->emission[i][n]);
		}
	}

	return;
}

void loadObserves(char* observeFile) {
	int i = 0;
	int o;
	fp = fopen(observeFile, "r");
	while(fscanf(fp, "%d ", &o) > 0) {
		observ[i] = o;
		i += 1;
	}
	observ_num = i;
	return ;
}

void count_Alpha() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(i = 0; i < N; i++) {
		if(emission_mask[i][observ[0]] > 0)
			Alpha[0][i] = model.initial[i] * model.emission[i][observ[0]];
		else
			Alpha[0][i] = 0.0;
	}

	for(t = 1; t < T; t++) {
		for(i = 0; i < N; i++) {
			if(emission_mask[i][observ[t]] > 0) {
				double tmpsum = 0.0;

				for(n = 0; n < N; n++) {
					tmpsum += Alpha[t-1][n] * model.transition[n][i];
				}

				Alpha[t][i] = tmpsum * model.emission[i][observ[t]];
			}
			else
				Alpha[t][i] = 0.0;
			
		}
		
	}

	return;
}

void count_Beta() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(i = 0; i < N; i++) {
		Beta[T-1][i] = 1.0;
	}

	for(t = T-2; t >= 0; t--) {

		for(i = 0; i < N; i++)
			Beta[t][i] = 0.0;

		for(n = 0; n < N; n++) {
			if(emission_mask[n][observ[t+1]] > 0)
				for(i = 0; i < N; i++) {
				
					
					Beta[t][i] += model.transition[i][n] * model.emission[n][observ[t+1]] * Beta[t+1][n];
				}
		}
	}

	return;
}

void count_Gamma() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(t = 0; t < T; t++) {
		double tmpsum = 0.0;
		for(i = 0; i < N; i++) {
			Gamma[t][i] = Alpha[t][i] * Beta[t][i];
			tmpsum += Gamma[t][i];
		}
		for(i = 0; i < N; i++) {
			if(tmpsum ==0)
				Gamma[t][i] = 0.0;
			else
				Gamma[t][i] /= tmpsum;
		}
	}


	return;
}

void count_Epsilon() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(t = 0; t < T - 1; t++) {

		double tmpsum = 0.0;
		for(n = 0; n < N; n++) {
			if(emission_mask[n][observ[t+1]] > 0)
				for(i = 0; i < N; i++) {

				

					Epsilon[t][i][n] = Alpha[t][i] * model.transition[i][n] 
											* model.emission[n][observ[t+1]] 
											* Beta[t+1][n];

					tmpsum += Epsilon[t][i][n];


				}
			else
				for(i = 0; i < N; i++)
					Epsilon[t][i][n] = 0.0;
		}
		for(i = 0; i < N; i++) {
			for(n = 0; n < N; n++) {
				if(tmpsum == 0)
					Epsilon[t][i][n] = 0.0;
				else
					Epsilon[t][i][n] /= tmpsum;
			}
		}
	}	

	return;
}


int get_observ() {
	int i = 0;
	int o;

	observ[i] = 36;
	i += 1;
	while(fscanf(fp, "%d ", &o) > 0) {
		observ[i] = o;
		i += 1;
		if(o == 36) { // o == ' '
			break;
		}
	}
	observ_num = i;
	return observ_num;
}

double get_logProb() {
	double tmpsum = 0.0;
	int N = model.state_num;
	int T = observ_num;
	int i;
	for(i = 0; i < N; i++) {
		tmpsum += Alpha[T-1][i];
	}

	if(tmpsum == 0.0)
		fprintf(stderr, "tmpsum in get_logProb is zero\n");

	return log(tmpsum);
}


int main(int argc, char* argv[]) {

	int i, j, k, t, m, n, o;
	int N, T;
	

	char* emissionFile = argv[1];
	char* observeFile = argv[2];

	char* dumpInitialFile = argv[3];
	char* dumpTransitionFile = argv[4];
	char* dumpEmissionFile = argv[5];

	int Iters = atoi(argv[6]);

	fprintf(stderr, "load model\n");
	loadModel(&model, emissionFile);

	
	N = model.state_num;
	T = observ_num;

	

	


	fprintf(stderr, "starting training\n");

	fp = fopen(observeFile, "r");

	clock_t t1, t2, total;
	total = clock();


	for(m = 0; m < Iters; m++) {
	
		fprintf(stderr,"****** iters: %d *********\n", m);
		t1 = clock();
		rewind(fp);

		// clear acc_matrice
		fprintf(stderr,"clearing\n");
		for(i = 0; i < N; i++)
		{
			new_initial[i] = 0;
			acc_gamma[i] = 0;
			acc_gamma_T[i] = 0;
			
			for(n = 0; n < N; n++){
				acc_epsilon[i][n] = 0;
				acc_gamma_observ[i][n] = 0;
			}
				
			
		}

		fprintf(stderr,"counting from observations\n");

		double logProb = 0.0;

		int p = 0;
		while(get_observ() > 1) {
			p += 1;
			//counting with observations
			// fprintf(stderr,"counting Alpha\n");
			count_Alpha();

			// if(observ_num == 30)
			// 	fprintf(stderr, "logporb: %lf\n", get_logProb());
			logProb += get_logProb();
			// fprintf(stderr,"counting Beta\n");
			count_Beta();
			// fprintf(stderr,"counting Gamma\n");
			count_Gamma();
			// fprintf(stderr,"counting Epsilon\n");
			count_Epsilon();

			// fprintf(stderr,"counting acc_matrice\n");

			N = model.state_num;
			T = observ_num;

			//counting acc_matrice
			for(i = 0; i < N; i++) {
				new_initial[i] += Gamma[0][i];
				for(t = 0; t < T; t++) {
					acc_gamma_T[i] += Gamma[t][i];
					if(t < T - 1)
						acc_gamma[i] += Gamma[t][i];

					acc_gamma_observ[i][observ[t]] += Gamma[t][i];
				
					if(t < T - 1)
						for(n = 0; n < N; n++) {
							acc_epsilon[i][n] += Epsilon[t][i][n];
						}


				}
			}


		}




		fprintf(stderr, "iter: %d, logProb = %lf\n", m, logProb);
		

		

		//updating
		fprintf(stderr,"updating\n");

		//initial
		for(i = 0; i < N; i++)
			model.initial[i] = new_initial[i] / p;


		//transition
		for(i = 0; i < N; i++) {
			if(acc_gamma[i] == 0) {
				double count = 0.0;
				for(n = 0; n < N; n++) {
					if(model.transition[i][n] != 0.0)
						count += 1.0;
				}

				for(n = 0; n < N; n++) {
					if(model.transition[i][n] != 0.0)
						model.transition[i][n] = 1.0 / count;
				}

			}
			else{
				for(n = 0; n < N; n++) 
					model.transition[i][n] = acc_epsilon[i][n] / acc_gamma[i];	
			}
			
		}

		//emission
		for(i = 0; i <N; i++) {
			if(acc_gamma_T[i] == 0.0) {
				double count = 0.0;
				for(n = 0; n < N; n++) {
					if(model.emission[i][n] != 0.0)
						count += 1.0;
				}

				for(n = 0; n < N; n++) {
					if(model.emission[i][n] != 0.0)
						model.emission[i][n] = 1.0 / count;
				}
			}
			else {
				for(n = 0; n < N; n++) 
					model.emission[i][n] = acc_gamma_observ[i][n] / acc_gamma_T[i];
				
			}
			
		}
		t2 = clock();
		fprintf(stderr, "time cost: %f\n",(double)(t2 - t1) / CLOCKS_PER_SEC);

		fprintf(stderr, "****** end of iteration %d *****\n", m);

		// char dumpInitial[100];
		// char dumpTrans[100];
		// char dumpEmission[100];
		// char iterstring[10];
		// sprintf(iterstring,"%d",m);
		// strcpy(dumpInitial, dumpInitialFile);
		// strcpy(dumpTrans, dumpTransitionFile);
		// strcpy(dumpEmission, dumpEmissionFile);
		// strcat(dumpInitial, iterstring);
		// strcat(dumpTrans, iterstring);
		// strcat(dumpEmission, iterstring);
		// dumpModel(&model, dumpInitial, dumpTrans, dumpEmission);

	}


	fprintf(stderr,"training completed\n");
	fprintf(stderr, "time cost: %f\n",(double)(clock() - total) / CLOCKS_PER_SEC);


	dumpModel(&model, dumpInitialFile, dumpTransitionFile, dumpEmissionFile);

	return 0;

}
