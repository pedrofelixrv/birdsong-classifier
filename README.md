# CLASSIFICAÇÃO DE AVES DO MUNICÍPIO DE ITAPAJÉ

Este projeto apresenta o desenvolvimento de um sistema de classificação automática de espécies de aves a partir de seus cantos, utilizando técnicas de processamento digital de sinais e aprendizado profundo. Inicialmente, os áudios são pré-processados e transformados em representações no domínio do tempo-frequência, como espectrogramas e matrizes de features, a fim de extrair características relevantes dos sinais sonoros. Em seguida, essas representações são utilizadas para treinar um modelo híbrido CNN-LSTM, que combina redes neurais convolucionais para extração espacial de padrões acústicos e redes recorrentes do tipo LSTM para modelar dependências temporais. O desempenho do modelo é avaliado por meio de métricas de classificação, demonstrando o potencial da abordagem para auxiliar na identificação automática de espécies, com aplicações em monitoramento ambiental e conservação da biodiversidade.

## Conteúdo deste Repositório

1. Relatório - (Detalha todo o processo e metodologia aplicada durante o desenvolvimento deste projeto como também as técnicas e a origem dos dados utilizados);
2. Scripts/Modelo - (Códigos utilizados para realizar processos (extração de dados, conversão para matrizes de features) e a arquitetura do modelo utilzado);
3. Data - (Arquivos .csv que contém o nome das espécies);
4. Requerimentos.txt - (Requerimentos necessários para executar em um ambiente controlado).

## Fluxograma
1. Extrair dados;
2. Pré-processamento;
3. Treinar modelo;
4. Executar Testes.
