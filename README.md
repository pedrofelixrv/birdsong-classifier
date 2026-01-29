# CLASSIFICAÇÃO DE AVES DO MUNICÍPIO DE ITAPAJÉ

Este projeto apresenta o desenvolvimento de um sistema de classificação automática de espécies de aves a partir de seus cantos, utilizando técnicas de processamento digital de sinais e aprendizado profundo. Inicialmente, os áudios são pré-processados e transformados em representações no domínio do tempo-frequência, como espectrogramas e matrizes de features, a fim de extrair características relevantes dos sinais sonoros. Em seguida, essas representações são utilizadas para treinar um modelo híbrido CNN-LSTM, que combina redes neurais convolucionais para extração espacial de padrões acústicos e redes recorrentes do tipo LSTM para modelar dependências temporais. O desempenho do modelo é avaliado por meio de métricas de classificação, demonstrando o potencial da abordagem para auxiliar na identificação automática de espécies, com aplicações em monitoramento ambiental e conservação da biodiversidade.

Palavras-chave: Processamento de sinais, aprendizado profundo, classificação de áudio, canto de aves, redes neurais convolucionais (CNN), LSTM, bioacústica, reconhecimento de padrões, espectrogramas.

Este projeto propôs o desenvolvimento de um sistema computacional capaz de classificar automaticamente espécies de aves a partir de seus cantos, utilizando técnicas de processamento digital de sinais e aprendizado profundo. A abordagem adotada consiste na conversão dos sinais de áudio em representações tempo-frequência e no treinamento de um modelo híbrido baseado em redes neurais convolucionais (CNN) e redes recorrentes do tipo LSTM.

A proposta visou explorar a capacidade dessas arquiteturas em extrair características acústicas relevantes e modelar a dinâmica temporal dos sinais sonoros, buscando obter um desempenho satisfatório mesmo diante de variações naturais dos cantos e da presença de ruídos ambientais.

## Conteúdo deste Repositório

1. Relatórios - (Detalha todo o processo e metodologia aplicada durante o desenvolvimento deste projeto como também as técnicas e a origem dos dados utilizados);
2. Scripts/Modelo - (Códigos utilizados para realizar processos (extração de dados, conversão para matrizes de features, predição com novos dados) e a arquitetura do modelo utilzado);
3. Datas - (Arquivo .csv que contém o nome das espécies e um áudio para teste);
4. Requerimentos.txt - (Requerimentos necessários para executar em um ambiente controlado).

## Fluxograma
1. Extrair dados;
2. Pré-processamento;
3. Treinar modelo;
4. Executar Testes.
