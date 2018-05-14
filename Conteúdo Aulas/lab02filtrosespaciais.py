
# coding: utf-8

# In[17]:

get_ipython().magic('matplotlib inline')


# ## Laboratório 02 filtragem espacial
# 
# Nesse laboratório vamos explorar alguns dos filtros espaciais mais comuns.
# 
# No bloco abaixo, definimos algumas funções utilitárias para mostrar imagens e gerar imagens ruidosas.

# In[18]:

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data
from skimage.util import random_noise
from skimage.filters import gaussian, median
from skimage.morphology import disk

# função de convolução bidimensional, vamos referenciar a função pelo
# apelido conv2
from scipy.signal import convolve2d as conv2

def mostra_imagem_cinza(data):
  """Uma função auxiliar para exibir uma imagem em níveis de cinza"""
  f = plt.figure()
  ax_img = f.gca()
  ax_img.imshow(data, cmap=plt.cm.gray)
  ax_img.set_axis_off()
  return (f, ax_img)

def mostra_histograma(data, num_particoes=256):
  """Uma função auxiliar para exibir o histograma de uma imagem
  
  num_particoes: número de partições do histograma 
  """
  f = plt.figure()
  ax_hist = f.gca()
  ax_hist.hist(data.ravel(), bins=num_particoes, histtype='step', color='black', linewidth=1.2)
  ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
  ax_hist.set_xlabel('Intensidade de pixel')
  return (f, ax_hist)
 
def ruido_gaussiano(imagem, variancia):
  """Adiciona ruido gaussiano a uma imagem.
  Note que a imagem devolvida é do tipo ponto flutuante.
  """
  return random_noise(imagem, mode='gaussian', seed=378, var=variancia)
  
def ruido_sal_pimenta(imagem, proporcao=0.01):
  """Adiciona ruido sal e pimenta a uma imagem.
  Note que a imagem devolvida é do tipo ponto flutuante.
  """
  return random_noise(imagem, mode='s&p', seed=378, amount=proporcao)  
  



# Primeiro vamos fazer a leitura da imagem de entrada ("astronauta"). Essa imagem é colorida originalmente, por isso vamos converte-lá para uma escala de cinza primeiro.

# In[19]:

imagem = color.rgb2gray(data.camera())

# primeiro vamos mostrar a imagem e seu histograma
mostra_imagem_cinza(imagem);
# o Jupyter notebook imprime o último valor calculado 
# em uma célula de código, para evitar isso terminar
# a linha de código com ; para suprimir a saída
mostra_histograma(imagem);




# Para estudar os efeitos dos filtros, vamos gerar algumas imagens com ruído adicionado.
# 
# Primeiro, vamos gerar uma imagem com ruído gaussiano. Para cada pixel da imagem de entrada, vamos somar um valor aleatório com distribuição gaussiana. A intensidade do ruído é controlada pela variância da distribuição gaussiana.
# 
# O trecho abaixo mostra como gerar uma imagem com ruído gaussiano. Nesse processo, a imagem gerada é implicitamente convertida para ponto flutuante e normalizada para o intervalo [0, 1]. Quando o ruído é adicionado, é possível que os valores caiam fora do intervalo. Na nossa implementação, valores são forçados para zero (se negativo) ou um (se positivo).
# 
# **Exercício:** No bloco de código abaixo, altere o valor do parâmetro ``variancia`` e observe o efeito sobre a imagem produzida.

# In[20]:

# versão da imagem com ruido gaussiano adicionado
# o parametro variancia controla a intensidade do ruido
imagem_gauss = ruido_gaussiano(imagem, variancia=0.04)
mostra_imagem_cinza(imagem_gauss);
mostra_histograma(imagem_gauss);


# Um outro tipo de ruído usual é o chamado de *sal e pimenta*, por que seu efeito é produzir pontos claros e escuros na imagem. Para criar uma imagem com esse tipo de ruído, seguimos os seguintes passos: (a) converter a imagem de saída para ponto flutuante, normalizando a imagem para o intervalo [0, 1]; (b) selecionar aleatoriamente uma quantidade de pixels da imagem de saída para receber o valor 1.0 (*sal*); (c) selecionar aleatoriamente uma quantidade de pixels da imagem de saída para receber o valor 0.0 (*pimenta*). O parâmetro ``proporcao`` controla a proporção total de pixels que são alterados (o número de pixels convertidos para zero é igual à quantidade de pixels convertida para um).
# 
# **Exercício:** No bloco abaixo, altere o valor do parâmetro ``proporcao`` e note seu efeito sobre a imagem produzida.

# In[21]:

imagem_sp = ruido_sal_pimenta(imagem, proporcao=0.02)
fsp, axsp = mostra_imagem_cinza(imagem_sp)
axsp.set_title('Imagem com ruido do tipo sal e pimenta')
fhsp, axhsp = mostra_histograma(imagem_sp)
axhsp.set_title('Histograma, Imagem com ruido do tipo sal e pimenta');


# Agora vamos fazer a convolução das imagens ruidosas com filtros de média de vários tamanhos: 3x3, 5x5 e 11x11.
# 
# **Exercício:** o código abaixo realiza filtragem da imagem com ruído gaussiano usando uma máscara de média de tamanho 3x3. Repita para máscaras de média de tamanho 5x5 e 11x11.

# In[22]:

# A funcao ones gera uma matriz de determinado tamanho, preenchida
# totalmente com 1s. É preciso dividir pelo número de elementos da
# matriz para normalizar a mascara do filtro
w3x3 = np.ones((3, 3), dtype=np.float32)/9

w9x9 = np.ones((9, 9), dtype=np.float32)/81

print('Máscara do filtro de média 3x3:')
print(w3x3)





# In[23]:

# A função convolve2d realiza a convolução bidimensional entre
# duas matrizes. Note que no início do program nós criamos o 
# 'apelido' conv2 para facilitar a digitação
# O parametro 'same' indica que a matriz devolvida deve
# ter o mesmo tamanho da image original
media3 = conv2(imagem_gauss, w3x3, 'same')

fm3, axm3 = mostra_imagem_cinza(media3)
axm3.set_title('Media 3x3, Imagem com ruido gaussiano');


# Se calcularmos a diferença entre a imagem ruidosa e a imagem suavizada com o filtro de média, podemos ter uma noção de quanto ruído foi retirado da imagem.

# In[24]:

diferenca_3x3 = imagem_gauss - media3
fd, axd = mostra_imagem_cinza(diferenca_3x3)
axd.set_title('Imagem diferenca, (imagem ruido gaussiano) - (media 3x3)');


# **Exercício:** Para a imagem com ruído sal e pimenta, repita os passos de filtragem com máscara de média 3x3, 5x5 e 11x11.
# 
# Nas duas imagens ruidosas, varie o nível de ruído gaussiano e a proporção de pixels alterados. Repita a filtragem. Observe o efeitos do nível de ruído de entrada e do tamanho da máscara sobre: eliminação de ruído na imagem de saída e preservação de detalhes na imagem de saída.

# ## Outros filtros de suavização
# 
# Uma possível desvantagem do filtro de média é que todos os pixels vistos sob a máscara tem o mesmo peso no cálculo da média. Às vezes isso pode resultar em perda excessiva da definição de contornos. Por isso, podemos usar pesos diferentes para os elementos da máscara, resultando em um filtro de média ponderada, como no exemplo abaixo.

# In[31]:

filtro_suavizacao = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=np.float32)/16
resultado = conv2(imagem_gauss, filtro_suavizacao, 'same')
f4, ax4 = mostra_imagem_cinza(resultado)
ax4.set_title('Filtro de suavizacao, uma aplicacao')

resultado2 = conv2(resultado, filtro_suavizacao, 'same')
f5, ax5 = mostra_imagem_cinza(resultado2)
ax5.set_title('Filtro de suavizacao, duas aplicacoes')

diferenca = imagem_gauss - resultado
f6, ax6 = mostra_imagem_cinza(diferenca)
ax6.set_title('Imagem diferenca, (imagem com ruido) - (imagem suavizada)');



# **Exercício:** Compare o filtro de suavização acima com os filtros de média simples. Qual deles se comporta melhor quanto a preservação dos contornos da imagem?
# 
# **Exercício:** Qual é o efeito de aplicar o mesmo filtro de suavização duas vezes em seguida?

# ## Filtro Gaussiano
# 
# Uma alternativa ao filtro de média é o *filtro gaussiano*. A máscara de um filtro gaussiano é uma aproximação de uma função gaussiana bidimensional: $w(x, y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$. Como a função gaussiana tem suporte sobre toda a reta real, na prática o filtro precisa ser truncado em um certo tamanho.
# 
# A biblioteca ``skimage`` já tem uma implementação de filtro gaussiano, demonstrada abaixo:

# In[26]:

resultado_sigma1 = gaussian(imagem_gauss, sigma=1)
fg, axg = mostra_imagem_cinza(resultado_sigma1)
axg.set_title('Filtro de suavizacao Gaussiano, sigma unitario');


# **Exercício:** Altere o valor do sigma do filtro gaussiano, observe seu efeito sobre a eliminação de ruído e sobre a distorção de contornos da imagem.
# 
# **Exercício:** Repita os passos acima, agora usando a imagem distorcida por ruído sal e pimenta. Altere o valor do sigma do filtro gaussiano, observe seu efeito sobre a eliminação de ruído e sobre a distorção de contornos da imagem.
# 
# **Exercício:** Compare qualitativamente os resultados obtidos com o filtro de média e com o filtro gaussiano.

# ### Filtro de Mediana
# 
# Uma desvantagem do filtro de média é que todos os valores *enxergados* por meio da janela do filtro entram no cálculo do valor de saída. Quando algum desses valores de entrada for muito discrepante (muito baixo ou alto) em relação aos outros, ele ainda terá grande influência no valor de saída. E quando a máscara encontra-se numa região de borda (isto é, uma região de descontinuidade) o valor de saída será uma espécie de interpolação entre os dois lados da borda, resultando em uma perda de definição dos contornos da imagem.
# 
# O *filtro de mediana* busca amenizar esse efeito de perde de definição. No filtro de mediana, em vez de calcularmos uma média dos valores de entrada, nós vamos: (i) ordenar os valores da imagem de entrada que são vistos pela janela e (ii) tomar o valor mediano dentro dessa região para colocar na imagem de saída. Como efeitos da filtragem de mediana, os valores extremos (muito baixos ou muito altos) de uma região não influem no valor de saída; nas regiões de borda, existe menos borramento.
# 
# Na biblioteca ``skimage``, o filtro de mediana é implementado pela função ``skimage.filter.median``. Para usar o filtro de mediana, devemos especificar a região sobre a qual a mediana será calculada.

# In[27]:

# dessa maneira, a mediana é calculada sobre um quadrado 3x3
median_3x3 = median(imagem_gauss)

fm3, axm3 = mostra_imagem_cinza(median_3x3)
axm3.set_title('Resultado da filtragem de mediana, regiao 3x3');





# In[28]:

# Nos podemos calcular a mediana usando uma região especificada como matriz de 
# zeros e uns. Abaixo, vamos usar uma região em formato de disco (discretizado)
d3 = disk(3)
print("Disco de raio 3")
print(d3)

median_d3 = median(imagem_gauss, selem=d3)

fd3, axd3 = mostra_imagem_cinza(median_d3)
axd3.set_title('Resultado da filtragem de mediana, disco de raio 3');


# **Exercício:** Usando o código acima como ponto de partida, calcule a mediana usando discos de raio progressivamente maior. Observe o efeito do raio do disco sobre: eliminação de ruído; preservação de detalhes da imagem; e sobre a definição de contornos da imagem produzida. Teste com diferentes níveis de ruído (variância).
# 
# Usando a função ``np.ones``, produza regiões retangulares de diferentes tamanhos para calcular a mediana. Note se o uso de regiões retangulares maiores produz efeitos visíveis na imagem de saída.
# 
# **Exercício:** Repita os passos que você fez acima para filtragem de mediana, agora usando a imagem distorcida por ruído sal e pimenta. Varie a proporção de pixels distorcidos pelo ruído.
# 
# **Exercício:** Compare qualitativamente os resultados obtidos com filtro de média e com filtro de mediana.
# 
# **Exercício:** Compare qualitativamente os resultados obtidos com filtro de gaussiano e com filtro de mediana.
# 
# **Exercício para casa:** Uma maneira de gerar uma imagem ruidosa é fotografar em condições de baixa luminosidade. Com uma câmera digital, tente tirar fotografias de uma mesma cena, em níveis variados de iluminação. Usando algum software de visualização de fotos, compare visualmente a qualidade das imagens. Aplique os filtros de média, gaussiano e de mediana sobre as suas imagens mais ruidosas. Analise os resultados qualitativamente.

# # Filtro Laplaciano
# 
# O filtro laplaciano é uma aproximação discreta do operador $\nabla^2f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$.
# 
# Uma das maneiras de implementar o laplaciano é pelo uso das máscaras mostradas no trecho abaixo:

# In[29]:

nabla2_cruz = np.array([[0,  -1,  0],
                        [-1,  4, -1],
                        [0,  -1,  0]], dtype=np.float32)
laplaciano1 = conv2(imagem, nabla2_cruz, 'same')
f7, ax7 = mostra_imagem_cinza(laplaciano1)
ax7.set_title('Resultado do Laplaciano (em cruz)')

nabla2_diag = np.array([[-1,  -1,  -1],
                        [-1,  8, -1],
                        [-1,  -1,  -1]], dtype=np.float32)


laplaciano2 = conv2(imagem, nabla2_diag, 'same')
f8, ax8 = mostra_imagem_cinza(laplaciano2)
ax8.set_title('Resultado do Laplaciano (com diagonal)')

f9, ax9 = mostra_imagem_cinza(np.abs(laplaciano2))
ax9.set_title('Valor Absoluto Resultado do Laplaciano (com diagonal)');



# **Exercício:** Aplique o operador laplaciano sobre a imagem corrompida com ruído gaussiano (você vai produzir dois resultados, cada um usando uma das máscaras definidas acima). Varie o nível de ruído da imagem. Tente explicar o resultado.
# 
# **Exercício:** Em imagens ruidosas, o resultado da aplicação do laplaciano pode ser melhorado se aplicarmos uma suavização prévia. Aplique o operador laplaciano a uma imagem suavizada com o filtro gaussiano. Escolha um nível de ruído que não resulte em degradação excessiva. Varie o sigma do filtro gaussiano em busca de um melhor resultado (isto é, contornos bem definidos e uma imagem resultante com pouco ruído).
# 
# **Exercício:** Um outro meio de se obter um resultado parecido com o do laplaciano é pela aplicação do operador *diferença de gaussianas*. Esse operador é definido como a diferença entre duas imagens, $g_1 - g_2$, em que $g_i$ é obtida a partir da imagem de entrada $f$ pela aplicação de um filtro gaussiano com $\sigma_i$; e $\sigma_2 > \sigma_1$ (isto é, $g_2$ é uma imagem mais borrada que $g_1$). Aplique o operador sobre a imagem corrompida com ruído gaussiano (escolha um valor fixo para o nível de ruído). Varie o valor de $\sigma_1, \sigma_2$ de modo a obter um bom resultado (uma heurística é tentar manter uma relação em torno de $\sigma_2 \approx 1.6\sigma_1$, tente também com $\sigma_2 \approx 4\sigma_1$). Compare com o resultado obtido com o Laplaciano.
