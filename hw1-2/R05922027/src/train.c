#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#define MAX_STATE 55
#define MAX_SEQ 3000010




typedef struct{

	int state_num; 
	double transition[MAX_STATE][MAX_STATE];
	double emission[MAX_STATE][MAX_STATE];
	double initial[MAX_STATE];

}Model;

Model model;

double **Alpha;//[MAX_SEQ][MAX_STATE];
double **Beta;//[MAX_SEQ][MAX_STATE];
double **Gamma;//[MAX_SEQ][MAX_STATE];
double ***Epsilon;//[MAX_SEQ][MAX_STATE][MAX_STATE];

double acc_gamma[MAX_STATE];
double acc_gamma_T[MAX_STATE];
double acc_gamma_observ[MAX_STATE][MAX_STATE];
double acc_epsilon[MAX_STATE][MAX_STATE];
double new_initial[MAX_STATE];

int observ[MAX_SEQ];
int observ_all[MAX_SEQ];
int observ_num = MAX_SEQ;
int observ_max = MAX_SEQ;
int observ_total = MAX_SEQ;
int observ_now = 0;

FILE * fp;

int emission_mask[MAX_STATE][MAX_STATE];

double LOWBOUND = -709.089565713;

double log1pexp(double x)
{   
	return x < LOWBOUND ? 0. : log1p(exp(x));
	// return x < -309.089565713 ? 0. : log1p(exp(x));

}

double logSum(double logA, double logB) {

	if(logA < LOWBOUND) return logB;
	else if(logB < LOWBOUND) return logA;
	else
		return logA > logB ? logA + log1pexp(logB - logA) : logB + log1pexp(logA - logB);
}

double logMul(double logA, double logB) {
	return logA + logB;
}

void loadModel(Model* m, char *emissionFile) {
	// load emission probs
	double emission_sum[MAX_STATE];
	int i, j;
	double b;
	fp = fopen(emissionFile, "r");

	m->state_num = 0;

	for(i = 0; i < MAX_STATE; i++)
		emission_sum[i] = -INFINITY;

	while(fscanf(fp, "%d %d %lf\n", &i, &j, &b) > 0) {
		if(b > 0) {
			m->emission[i][j] = log(b);
			emission_mask[i][j] = 1;
		}
		else {
			m->emission[i][j] = -INFINITY;
			emission_mask[i][j] = 0;
		}
		if(emission_mask[i][j] > 0)
			emission_sum[i] = logSum(emission_sum[i], m->emission[i][j]);
		if(m->state_num < i+1)
			m->state_num = i+1;
		if(m->state_num < j+1)
			m->state_num = j+1;
	}

	// normalize
	for(i = 0; i < m->state_num; i++) {
		for(j = 0; j < m->state_num; j++) {
			// m->emission[i][j] /= emission_sum[i];
			if(emission_mask[i][j] > 0)
				m->emission[i][j] -= emission_sum[i];
		}
	}

	// uniformly assign transition probs and initial probs

	for(i = 0; i < m->state_num; i++) {
		m->initial[i] = -log(m->state_num);
		for(j = 0; j < m->state_num; j++) {
			// m->transition[i][j] = 1.0 / m->state_num;
			m->transition[i][j] = -log(m->state_num);
		}
	}
	fprintf(stderr, "%d\n", model.state_num);
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
	observ_total = 0;
	int max = 0;
	int space = 0;
	int spaceId = model.state_num - 1;
	fp = fopen(observeFile, "r");
	while(fscanf(fp, "%d ", &o) > 0) {
		observ_all[i] = o;
		i += 1;
		if(o == spaceId) {
			if(max < i - space) {
				max = i - space;
			}
			space = i;
		}
	}
	observ_total = i;
	observ_max = max + 1;
	// printf("max: %d\n", max+);
	fclose(fp);

	return ;
}

void MallocMatrix() {

	Alpha = (double**)malloc(observ_max * sizeof(double*));
	Beta = (double**)malloc(observ_max * sizeof(double*));
	Gamma = (double**)malloc(observ_max * sizeof(double*));
	Epsilon = (double***)malloc(observ_max * sizeof(double**));

	int i, t;
	int T = observ_max;
	int N = model.state_num;

	for(t = 0; t < T; t++) {
		Alpha[t] = (double*)malloc(N * sizeof(double));
		Beta[t] = (double*)malloc(N * sizeof(double));
		Gamma[t] = (double*)malloc(N * sizeof(double));
		Epsilon[t] = (double**)malloc(N * sizeof(double*));
		for(i = 0; i < N; i++) {
			Epsilon[t][i] = (double*)malloc(N * sizeof(double));
		}
	}

	return;

}

void count_Alpha() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	
	for(i = 0; i < N; i++) {
		if(emission_mask[i][observ[0]] > 0)
			Alpha[0][i] = model.initial[i] + model.emission[i][observ[0]];
		else
			Alpha[0][i] = -INFINITY;
	}

	for(t = 1; t < T; t++) {
		for(i = 0; i < N; i++) {
			

			if(emission_mask[i][observ[t]] > 0) {
				double tmpsum = -INFINITY;

				for(n = 0; n < N; n++) {
					
						tmpsum = logSum(tmpsum, Alpha[t-1][n] + model.transition[n][i]);

				}

				Alpha[t][i] = tmpsum + model.emission[i][observ[t]];
			}
			else {
				Alpha[t][i] = -INFINITY;
			}


			
		}
		
	}

	return;
}

void count_Beta() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(i = 0; i < N; i++) {
		Beta[T-1][i] = log(1.0);
	}


	for(t = T-2; t >= 0; t--) {

		for(i = 0; i < N; i++)
			Beta[t][i] = -INFINITY;
			
		for(n = 0; n < N; n++) {
			if(emission_mask[n][observ[t+1]] > 0){
				
				for(i = 0; i < N; i++) {	
					Beta[t][i] = logSum(Beta[t][i], model.transition[i][n] + model.emission[n][observ[t+1]] + Beta[t+1][n]);

				}
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
		double tmpsum = -INFINITY;
		for(i = 0; i < N; i++) {
			
				Gamma[t][i] = Alpha[t][i] + Beta[t][i];
				tmpsum = logSum(tmpsum, Gamma[t][i]);
			
			
		}
		for(i = 0; i < N; i++) {
			
					Gamma[t][i] = Gamma[t][i] - tmpsum;

					if(t == 0) {
						new_initial[i] = logSum(new_initial[i], Gamma[0][i]);
					}
					acc_gamma_T[i] = logSum(acc_gamma_T[i], Gamma[t][i]);
					if(t < T - 1)
						acc_gamma[i] = logSum(acc_gamma[i], Gamma[t][i]);
					acc_gamma_observ[i][observ[t]] = logSum(acc_gamma_observ[i][observ[t]], Gamma[t][i]);

				
				

			
				
		}
	}


	return;
}

void count_Epsilon() {

	int n, t, i;
	int N = model.state_num;
	int T = observ_num;

	for(t = 0; t < T - 1; t++) {
		// printf("t: %d, T: %d, tmax:%d\n", t, T, observ_max);

		double tmpsum = -INFINITY;
		

		for(n = 0; n < N; n++) {
			if(emission_mask[n][observ[t+1]] > 0){
				for(i = 0; i < N; i++) {

					Epsilon[t][i][n] = Alpha[t][i] + model.transition[i][n] 
											+ model.emission[n][observ[t+1]] 
											+ Beta[t+1][n];

					tmpsum = logSum(tmpsum, Epsilon[t][i][n]);


				}
			}
			else {
				for(i = 0; i < N; i++) {
					Epsilon[t][i][n] = -INFINITY;
				}
			}

			
		}
		for(i = 0; i < N; i++) {
			for(n = 0; n < N; n++) {
				
					Epsilon[t][i][n] -= tmpsum;
					acc_epsilon[i][n] = logSum(acc_epsilon[i][n], Epsilon[t][i][n]);
				
					
			}
		}
	}	

	return;
}


int get_observ() {
	int i = 0;
	int o;

	if(observ_now >= observ_total) {
		observ_now = 0;
		return 0;
	}
		

	observ[i] = model.state_num - 1; // means ' '
	i += 1;
	while(observ_now < observ_total) {
		o = observ_all[observ_now];
		observ_now += 1;
		observ[i] = o;
		i += 1;
		if(o == model.state_num - 1) { // o == ' '
			// fprintf(stderr, "%d, %d: break\n", observ_now, observ_total);
			break;
		}
		// fprintf(stderr, "%d, %d\n", observ_now, observ_total);
		
	}
	
	observ_num = i;
	// fprintf(stderr, "observ: %d\n", observ_num);
	return observ_num;
}

double get_logProb() {
	double tmpsum = -INFINITY;
	int N = model.state_num;
	int T = observ_num;
	int i;
	tmpsum = Alpha[T-1][0];
	for(i = 1; i < N; i++) {
		tmpsum = logSum(tmpsum, Alpha[T-1][i]);
		// fprintf(stderr, "%lf %lf\n", Alpha[T-1][i], tmpsum);
	}
	// fprintf(stderr, ">>> %lf\n", tmpsum);


	// BUG ! because log0, prob is too low
	if(tmpsum == 0.0)
		fprintf(stderr, "tmpsum in get_logProb is zero\n");

	return tmpsum;
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
	loadObserves(observeFile);
	MallocMatrix();

	
	N = model.state_num;
	T = observ_max;


	// memset(Alpha, -INFINITY, sizeof(Alpha[0][0]) * T * N);
	// memset(Beta, -INFINITY, sizeof(Beta[0][0]) * T * N);
	// memset(Gamma, -INFINITY, sizeof(Gamma[0][0]) * T * N);
	// memset(Epsilon, -INFINITY, sizeof(Epsilon[0][0][0]) * T * N * N);




	double prev = -INFINITY;
	


	fprintf(stderr, "starting training\n");

	// fp = fopen(observeFile, "r");

	clock_t t1, t2, total;
	total = clock();


	for(m = 0; m < Iters; m++) {
	
		fprintf(stderr,"****** iters: %d *********\n", m);
		t1 = clock();
		// rewind(fp);
		observ_now = 0;

		// clear acc_matrice
		fprintf(stderr,"clearing\n");
		for(i = 0; i < N; i++)
		{
			new_initial[i] = -INFINITY;
			acc_gamma[i] = -INFINITY;
			acc_gamma_T[i] = -INFINITY;
			
			for(n = 0; n < N; n++){
				acc_epsilon[i][n] = -INFINITY;
				acc_gamma_observ[i][n] = -INFINITY;
			}
				
			
		}
		N = model.state_num;
		// memset(new_initial, -INFINITY, sizeof(new_initial[0]) * N);
		// memset(acc_gamma, -INFINITY, sizeof(acc_gamma[0]) * N);
		// memset(acc_gamma_T, -INFINITY, sizeof(acc_gamma_T[0]) * N);
		// memset(acc_epsilon, -INFINITY, sizeof(acc_epsilon[0][0]) * N * N);
		// memset(acc_gamma_observ, -INFINITY, sizeof(acc_gamma_observ[0][0]) * N * N);

		fprintf(stderr,"counting from observations\n");

		double logProb = 0.0;

		int c = 0;

		int p = 0;
		while(get_observ() > 1) {
			p += 1;


			

			// c += observ_num-1;
			// fprintf(stderr, "%d/%d\n", c, observ_total);
			
			//counting with observations
			// fprintf(stderr,"counting Alpha\n");
			count_Alpha();

			logProb += get_logProb();
			// fprintf(stderr,"counting Beta\n");
			count_Beta();
			// fprintf(stderr,"counting Gamma\n");
			count_Gamma();
			// fprintf(stderr,"counting Epsilon\n");
			count_Epsilon();


			N = model.state_num;
			T = observ_num;

			// //counting acc_matrice
			// for(i = 0; i < N; i++) {
			// 	new_initial[i] = logSum(new_initial[i], Gamma[0][i]);
			// 	for(t = 0; t < T; t++) {
			// 		acc_gamma_T[i] = logSum(acc_gamma_T[i], Gamma[t][i]);
			// 		if(t < T - 1)
			// 			acc_gamma[i] = logSum(acc_gamma[i], Gamma[t][i]);

			// 		acc_gamma_observ[i][observ[t]] = logSum(acc_gamma_observ[i][observ[t]], Gamma[t][i]);
				
			// 		if(t < T - 1)
			// 			for(n = 0; n < N; n++) {
			// 				acc_epsilon[i][n] = logSum(acc_epsilon[i][n], Epsilon[t][i][n]);
			// 			}


			// 	}
			// }

			// memset(Alpha, -INFINITY, sizeof(Alpha[0][0]) * T * N);
			// memset(Beta, -INFINITY, sizeof(Beta[0][0]) * T * N);
			// memset(Gamma, -INFINITY, sizeof(Gamma[0][0]) * T * N);
			// memset(Epsilon, -INFINITY, sizeof(Epsilon[0][0][0]) * T * N * N);



		}






		fprintf(stderr, "iter: %d, logProb = %lf\n", m, logProb);

		if(logProb - prev < 1000.0)
			break;
		else
			prev = logProb;
		

		

		//updating
		fprintf(stderr,"updating\n");

		//initial
		for(i = 0; i < N; i++)
			model.initial[i] = new_initial[i] - log(p);

		//transition
		for(i = 0; i < N; i++) {

			for(n = 0; n < N; n++) 
				model.transition[i][n] = acc_epsilon[i][n] - acc_gamma[i];	
		
		
		}

		//emission
		for(i = 0; i <N; i++) {
			
			for(n = 0; n < N; n++) 
				model.emission[i][n] = acc_gamma_observ[i][n] - acc_gamma_T[i];
			
		}


		// //transition
		// for(i = 0; i < N; i++) {
		// 	if(acc_gamma[i] < LOWBOUND) {
		// 		double count = 0.0;
		// 		for(n = 0; n < N; n++) {
		// 			if(model.transition[i][n] >= LOWBOUND)
		// 				count += 1.0;
		// 		}

		// 		for(n = 0; n < N; n++) {
		// 			if(model.transition[i][n] >= LOWBOUND)
		// 				model.transition[i][n] = -log(count);
		// 		}

		// 	}
		// 	else{
		// 		for(n = 0; n < N; n++) 
		// 			model.transition[i][n] = acc_epsilon[i][n] - acc_gamma[i];	
		// 	}
			
		// }

		// //emission
		// for(i = 0; i <N; i++) {
		// 	if(acc_gamma_T[i] < LOWBOUND) {
		// 		double count = 0.0;
		// 		for(n = 0; n < N; n++) {
		// 			if(model.emission[i][n] >= LOWBOUND)
		// 				count += 1.0;
		// 		}

		// 		for(n = 0; n < N; n++) {
		// 			if(model.emission[i][n] >= LOWBOUND)
		// 				model.emission[i][n] = -log(count);
		// 		}
		// 	}
		// 	else {
		// 		for(n = 0; n < N; n++) 
		// 			model.emission[i][n] = acc_gamma_observ[i][n] - acc_gamma_T[i];
				
		// 	}
			
		// }

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
