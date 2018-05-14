# Análise Quantitativa de Materiais Bifásicos

## Sobre
Esse aplicativo foi desenvolvido como atividade prática para a aula de 
**Processamento Digital de Imagens** do curso de **Bacharelado em Ciência
Da Computação da UFABC** em Maio/2018.

## Resumo
O aplicativo tem como objetivo análisar micrografias a fim de quantificar
suas fases através da segmentação da imagem por cores. As micrografias
devem apresentar apenas duas fase, distintas visualmente uma da outra,
e será utilizada segmentação de imagem por limiar para obter a quanti-
dade de pixels de cada fase presentes na micrografia.

### Entradas:
*Micrografia com apenas duas fases
*Nome da Fase Clara
*Nome da Fase Escura
*Valor Limiar
*Bordas

### Saída:
* Percentual de cada fase na amostra.

## Algoritmo
1. A aplicação desenvolvida transforma as imagens para tons de cinza
2. Analisa cada pixel da imagem de acordo com o valor limiar:
2.1 Marca o pixel como 1 se o pixel for mais claro que o limiar.
2.2 Marca o pixel como 0 se o pixel for mais escuro que o limiar.
3. Conta quantos pixels 1s e 0s existem.
4. Apresenta o percentual de 1s e 0s para o usuário.


## Melhorias Necessárias:
Ou, o que não foi possível fazer no tempo do projeto:
* Incluir filtro para eliminação de ruído.
* Permitir a análise de imagens com mais de uma fase.
* Automatizar a obtenção do limiar ao invés de solicitar um valor ao usuário.
 