# Aprendizado Federado na Prática para Triagem do Câncer do Colo do Útero

Este projeto apresenta uma implementação prática de **Aprendizado Federado (Federated Learning)** aplicado à classificação de imagens citopatológicas do exame de Papanicolau.

A solução foi desenvolvida com o objetivo de demonstrar como múltiplas instituições podem colaborar no treinamento de modelos de Inteligência Artificial sem compartilhar dados sensíveis, preservando a privacidade dos pacientes e atendendo requisitos regulatórios como a LGPD.

O ambiente simula um cenário multicêntrico utilizando múltiplos clientes federados e um servidor central responsável pela agregação dos modelos.

---

# Objetivos

* Demonstrar os conceitos fundamentais de Aprendizado Federado;
* Implementar o algoritmo **Federated Averaging (FedAvg)**;
* Utilizar **Redes Neurais Convolucionais (CNNs)** para classificação de imagens citopatológicas;
* Simular múltiplas instituições utilizando contêineres Docker;
* Preservar a privacidade dos dados durante o treinamento;
* Avaliar o desempenho de modelos globais treinados de forma colaborativa.

---

# Arquitetura do Sistema

```text
                +------------------+
                |   Servidor FL    |
                |     Flower       |
                +--------+---------+
                         ^
                         |
        -------------------------------------
        |                 |                 |
        |                 |                 |
+-------+------+ +--------+------+ +--------+------+
| Cliente 1    | | Cliente 2     | | Cliente 3     |
| CNN PyTorch  | | CNN PyTorch   | | CNN PyTorch   |
| Dados Locais | | Dados Locais  | | Dados Locais  |
+--------------+ +---------------+ +---------------+
```

Cada cliente possui seu próprio conjunto de dados e realiza treinamento local.

Apenas os parâmetros da rede neural são enviados ao servidor para agregação.

Nenhuma imagem é compartilhada entre as instituições.

---

# Tecnologias Utilizadas

## Docker

Utilizado para criar ambientes isolados e reprodutíveis para clientes e servidor.

Benefícios:

* facilidade de implantação;
* reprodutibilidade experimental;
* isolamento entre componentes;
* simulação de múltiplas instituições.

---

## Flower

Framework responsável pela infraestrutura de Aprendizado Federado.

Principais funcionalidades:

* comunicação cliente-servidor;
* gerenciamento das rodadas federadas;
* implementação do algoritmo FedAvg;
* agregação dos modelos locais.

---

## PyTorch

Framework utilizado para construção e treinamento das Redes Neurais Convolucionais.

Recursos utilizados:

* definição das arquiteturas CNN;
* treinamento e validação;
* otimização por gradiente;
* inferência de imagens.

---

## Torchvision

Biblioteca complementar do PyTorch utilizada para:

* transformações de imagens;
* carregamento de datasets;
* utilização de modelos pré-treinados.

---

# Banco de Dados

Este projeto **não utiliza banco de dados relacional ou NoSQL**.

Os dados são armazenados diretamente no sistema de arquivos em diretórios organizados por cliente.

Estrutura simplificada:

```text
data_sbcas/
|
├── data/
│   ├── client_1/
│   ├── client_2/
│   ├── client_3/
│   └── global_test/
```

Cada cliente possui acesso apenas ao seu diretório local.

O conjunto `global_test` é utilizado exclusivamente pelo servidor para avaliação do modelo global.

---

# Dataset

O projeto utiliza o dataset:

**Single Cell Conventional Pap Smear Images**

O conjunto contém imagens citológicas obtidas a partir de exames de Papanicolau e é utilizado para treinamento e avaliação dos modelos de classificação.

---

# Estrutura do Projeto

```text
project/
|
├── client/
│   ├── app.py
│   ├── model.py
│   ├── dataset.py
│   └── requirements.txt
|
├── server/
│   ├── app.py
│   ├── strategy.py
│   └── requirements.txt
|
├── data_sbcas/
│   └── data/
│       ├── client_1/
│       ├── client_2/
│       ├── client_3/
│       └── global_test/
|
├── docker-compose.yaml
├── README.md
└── LICENSE
```

---

# Dependências Principais

```text
flwr==1.8.0
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
pillow==10.3.0
```

---

# Executando o Projeto

## Construção dos Contêineres

```bash
docker-compose build
```

## Inicialização

```bash
docker-compose up
```

Durante a execução:

1. O servidor federado é iniciado;
2. Os clientes carregam seus dados locais;
3. O treinamento local é realizado;
4. Os pesos são enviados ao servidor;
5. O servidor executa a agregação FedAvg;
6. Um novo modelo global é distribuído;
7. O processo é repetido por múltiplas rodadas.

---

# Conceitos Demonstrados

* Aprendizado Federado (Federated Learning);
* Federated Averaging (FedAvg);
* Redes Neurais Convolucionais (CNN);
* Transfer Learning;
* Fine-Tuning;
* Docker;
* Flower;
* Treinamento Distribuído;
* Privacidade de Dados em Saúde;
* Inteligência Artificial Aplicada à Citopatologia.

---

# Referência

Este projeto foi desenvolvido como material prático do capítulo:

**Aprendizado Federado na Prática: Da Teoria à Implementação na Triagem do Câncer do Colo do Útero**

SBCAS – Simpósio Brasileiro de Computação Aplicada à Saúde.
