
# Importando as bibliotecas
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, html, dcc

# Lendo o arquivo
try:
    df = pd.read_csv(
        'C:/Users/costa/OneDrive/Documentos/EBAC/VSCODE/documentos_baixados_ebac/ecommerce_estatistica.csv'
    )
    print('Arquivo carregado com sucesso!')

except FileNotFoundError:
    print('Erro: arquivo CSV não encontrado.')
    raise SystemExit

# Visualizando as primeiras linhas do DataFrame
print(df.head().to_string())

# Visualizando os nomes das colunas
print(df.columns.tolist())

# Análise exploratória dos dados

# Verificando informações sobre as colunas e os tipos de dados
df.info()

# Visualizando estatísticas descritivas das variáveis numéricas
print(df.describe())

# Verificando a quantidade de valores ausentes em cada coluna
print(df.isnull().sum())

# Definindo a paleta de cores
cores = px.colors.qualitative.Plotly

##################################
###---- inicio dos graficos----###
##################################

#### 1. Gráfico de Histograma
# Objetivo: analisar a distribuição das notas dos produtos

fig1 = px.histogram(
    df,
    x='Nota',
    nbins=10,
    title='Distribuição das Notas dos Produtos',
    labels={'Nota': 'Nota dos Produtos'},
    color_discrete_sequence=cores
)

fig1.update_yaxes(title_text='Quantidade de Produtos')


#### 2. Gráfico de Dispersão
# Objetivo: analisar a relação entre preço e quantidade vendida

fig2 = px.scatter(
    df,
    x='Preço',
    y='Qtd_Vendidos_Cod',
    title='Relação entre Preço e Quantidade Vendida',
    labels={
        'Preço': 'Preço (R$)',
        'Qtd_Vendidos_Cod': 'Quantidade Vendida (valor codificado)'
    },
    color_discrete_sequence=cores
)


#### 3. Mapa de Calor
# Objetivo: identificar correlações entre variáveis numéricas

# Selecionando as colunas numéricas
colunas_numericas = [
    'Nota',
    'N_Avaliações',
    'Desconto',
    'Preço',
    'Qtd_Vendidos_Cod'
]

# Calculando a correlação entre as variáveis
correlacao = df[colunas_numericas].corr()

# Criando o gráfico
fig3 = px.imshow(
    correlacao,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    title='Mapa de Calor das Correlações',
    labels={'color': 'Correlação'}
)


#### 4. Gráfico de Barras
# Objetivo: identificar as 10 marcas com mais produtos

top_marcas = df['Marca'].value_counts().head(10)

fig4 = px.bar(
    x=top_marcas.values,
    y=top_marcas.index,
    orientation='h',
    title='As 10 Marcas com Mais Produtos',
    labels={
        'x': 'Quantidade de Produtos',
        'y': 'Marca'
    },
    color_discrete_sequence=cores
)


#### 5. Gráfico de Pizza
# Objetivo: visualizar a proporção dos produtos por gênero

# Contabilizando os produtos por gênero
genero = df['Gênero'].value_counts()

# Agrupando categorias com menos de 3% em Outros
genero_principal = genero[genero / genero.sum() >= 0.03]

outros = genero[genero / genero.sum() < 0.03].sum()

genero_agrupado = pd.concat([
    genero_principal,
    pd.Series({'Outros': outros})
])

# Criando o gráfico
fig5 = px.pie(
    names=genero_agrupado.index,
    values=genero_agrupado.values,
    title='Distribuição dos Produtos por Gênero',
    color_discrete_sequence=cores
)

fig5.update_traces(
    textposition='inside',
    textinfo='percent'
)


#### 6. Gráfico de Densidade
# Objetivo: analisar a distribuição das notas dos produtos

# Calculando a densidade das notas
from scipy.stats import gaussian_kde

notas = df['Nota'].dropna()

densidade = gaussian_kde(notas)

x = np.linspace(notas.min(), notas.max(), 200)
y = densidade(x)

# Criando o gráfico
fig6 = px.area(
    x=x,
    y=y,
    title='Densidade das Notas dos Produtos',
    labels={
        'x': 'Nota dos Produtos',
        'y': 'Densidade'
    },
    color_discrete_sequence=cores
)


#### 7. Gráfico de Regressão
# Objetivo: analisar a relação entre avaliações e quantidade vendida

# Selecionando os dados válidos
dados = df[['N_Avaliações', 'Qtd_Vendidos_Cod']].dropna()

# Calculando a regressão linear
coeficientes = np.polyfit(
    dados['N_Avaliações'],
    dados['Qtd_Vendidos_Cod'],
    1
)

# Criando o gráfico
fig7 = px.scatter(
    dados,
    x='N_Avaliações',
    y='Qtd_Vendidos_Cod',
    title='Regressão entre Avaliações e Quantidade Vendida',
    labels={
        'N_Avaliações': 'Número de Avaliações',
        'Qtd_Vendidos_Cod': 'Quantidade Vendida (valor codificado)'
    },
    color_discrete_sequence=cores
)

# Adicionando a linha de regressão
x_linha = np.linspace(
    dados['N_Avaliações'].min(),
    dados['N_Avaliações'].max(),
    100
)

y_linha = np.polyval(coeficientes, x_linha)

fig7.add_trace(
    go.Scatter(
        x=x_linha,
        y=y_linha,
        mode='lines',
        name='Regressão Linear',
        line=dict(color='red')
    )
)

##################################
###----Término dos graficos----###
##################################


##################################
### Criando a aplicação Dash  ####
##################################

app = Dash(__name__)

# Organizando os gráficos na aplicação
app.layout = html.Div([
    html.H1(
        'Análise Estatística do E-commerce',
        style={'textAlign': 'center'}
    ),

    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig6),
    dcc.Graph(figure=fig7)
])

# Executando a aplicação
if __name__ == '__main__':
    app.run(debug=False)

