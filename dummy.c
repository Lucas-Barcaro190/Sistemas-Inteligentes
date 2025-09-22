#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
	char ambiente[2], limpeza[2];
	ambiente[0] = 'a';
	ambiente[1] = 'b';
	limpeza[0] = 's';
	limpeza[1] = 's';
	char ambienteAtual = 'i';
	while(true){
		if(ambienteAtual == 'a' && limpeza[0] == 's'){
			printf("Sugando A\n");
			limpeza[0] = 'l';
		}
		else{
		if(ambienteAtual == 'a' && limpeza[0] != 's'){
			printf("Ambiente limpo, indo para a direita\n");
			ambienteAtual = 'b';
		}
		else{
			if(ambienteAtual == 'b' && limpeza[1] == 's'){
				printf("Sugando B\n");
				limpeza[1] = 'l';
		}
		else{
		if(ambienteAtual == 'b' && limpeza[1] != 's'){
			printf("Ambiente limpo, indo para a esquerda\n");
			ambienteAtual = 'a';
		}}}}

		printf("=================\n");
		printf("Estado atual: \n");
		printf("Aspirador de po em: ");
		putchar(ambienteAtual);
		printf("\nAmbiente A: ");
		putchar(limpeza[0]);
		printf("\nAmbiente B: ");
		putchar(limpeza[1]);
		printf("\n=================\n");
		sleep(2);
	}
	return 0;	
}
