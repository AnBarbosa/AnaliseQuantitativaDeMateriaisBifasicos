# An�lise Quantitativa de Materiais Bif�sicos

## Sobre
Esse aplicativo foi desenvolvido como atividade pr�tica para a aula de 
**Processamento Digital de Imagens** do curso de **Bacharelado em Ci�ncia
Da Computa��o da UFABC** em Maio/2018.

## Resumo
O aplicativo tem como objetivo an�lisar micrografias a fim de quantificar
suas fases atrav�s da segmenta��o da imagem por cores. As micrografias
devem apresentar apenas duas fase, distintas visualmente uma da outra,
e ser� utilizada segmenta��o de imagem por limiar para obter a quanti-
dade de pixels de cada fase presentes na micrografia.

### Entradas:
* Micrografia com apenas duas fases
* Nome da Fase Clara
* Nome da Fase Escura
* Valor Limiar
* Bordas

### Sa�da:
* Percentual de cada fase na amostra.

## Algoritmo
1. A aplica��o desenvolvida transforma as imagens para tons de cinza
2. Analisa cada pixel da imagem de acordo com o valor limiar:
2.1 Marca o pixel como 1 se o pixel for mais claro que o limiar.
2.2 Marca o pixel como 0 se o pixel for mais escuro que o limiar.
3. Conta quantos pixels 1s e 0s existem.
4. Apresenta o percentual de 1s e 0s para o usu�rio.


## Melhorias Necess�rias:
Ou, o que n�o foi poss�vel fazer no tempo do projeto:
* Incluir filtro para elimina��o de ru�do.
* Permitir a an�lise de imagens com mais de uma fase.
* Automatizar a obten��o do limiar ao inv�s de solicitar um valor ao usu�rio.
 