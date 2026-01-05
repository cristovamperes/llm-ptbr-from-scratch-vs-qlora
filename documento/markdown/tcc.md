> Nota: este arquivo é uma conversão automática do LaTeX. Citações foram convertidas para chaves BibTeX entre colchetes (ex.: [ross2023lost]). Para referências completas e formatação final, consulte o PDF.

O domínio sobre modelos de linguagem de grande escala (LLMs) deixou de ser uma questão puramente técnica para se tornar um vetor central de poder geopolítico, econômico e informacional. Nações e corporações que lideram o desenvolvimento de modelos como o GPT da OpenAI (ChatGPT), Claude e Gemini não apenas impulsionam a fronteira da inovação, mas também moldam a forma como milhões de pessoas acessam e interpretam informações. Essa concentração de poder levanta questões críticas sobre soberania digital, segurança informacional e a transferência implícita de valores culturais e epistemológicos [ross2023lost].

Diante deste cenário, a criação de um LLM nacional para o Brasil surge como uma necessidade estratégica. Contudo, os custos e desafios técnicos para treinar um modelo de ponta do zero são proibitivos para a maioria dos países. Este trabalho argumenta que existe um caminho alternativo e viável: o pós-treinamento e a adaptação de modelos de código aberto robustos. Em vez de apenas teorizar sobre a importância de um LLM brasileiro, este projeto se propõe a desenvolver um protótipo funcional (MVP/PoC), demonstrando um caminho prático para a capacitação tecnológica nacional.

A corrida pela inteligência artificial, sobretudo pelo controle de LLMs, determinará a relevância estratégica das nações nas próximas décadas. Modelos de fundação são hoje considerados infraestruturas críticas, com efeitos sistêmicos sobre a economia, a ciência e as políticas públicas [bommasani2021foundation]. A dependência de tecnologias estrangeiras cria uma vulnerabilidade estratégica, submetendo o país a decisões e vieses de modelos treinados em contextos socioculturais distintos.

Além disso, vivemos uma era de guerra informacional, onde o controle sobre os dados e os algoritmos que os processam é crucial. A capacidade de um país de desenvolver, alinhar e auditar seus próprios modelos de linguagem é fundamental para proteger dados sensíveis, garantir a integridade de processos democráticos e promover uma inovação alinhada às necessidades locais, refletindo os valores, a legislação e a diversidade cultural brasileira [bai2023alignment].

Contudo, a concretização de tal capacidade nacional enfrenta desafios técnicos consideráveis. A evolução dos LLMs, por exemplo, foi acelerada pela arquitetura Transformer [vaswani2017attention], que permitiu um salto em escala e capacidade de compreensão contextual. No entanto, o treinamento de modelos de fronteira demanda investimentos massivos em poder computacional, com custos que podem chegar a milhões de dólares [llmcosts2023]. Empresas como a NVIDIA se tornaram centrais neste ecossistema, desenvolvendo chips especializados (GPUs) que são essenciais para o treinamento. Esse hardware de ponta, concentrado em poucas nações, principalmente EUA e China, torna o desenvolvimento de um modelo do zero um desafio de longo prazo.

Felizmente, o ecossistema de código aberto oferece uma alternativa estratégica. Iniciativas como LLaMA [touvron2023llama] e Qwen [qwen2024team] disponibilizaram modelos de alta performance que podem servir como base para adaptação. Essa abordagem reduz drasticamente os custos e permite focar esforços em etapas de maior valor agregado, como a especialização do modelo para o contexto brasileiro.

Diante desse panorama de desafios e oportunidades, este trabalho propõe uma abordagem prática para avaliar a viabilidade de um LLM nacional, que se desdobra em duas frentes principais. Primeiramente, realiza uma **caracterização prática** de custos e requisitos (VRAM, tempo e custo estimado) para o treinamento do zero em escala reduzida, sustentada por logs e medições ao longo das versões (v1--v6). Em segundo lugar, e como foco principal, desenvolve evidência empírica de uma alternativa mais viável: a adaptação de modelos de código aberto por pós-treinamento eficiente (CPT/SFT via LoRA/QLoRA).

A metodologia de implementação se organiza em duas trilhas complementares, conectadas pela mesma restrição prática de hardware e tempo de treinamento:

    \item **Linha experimental (v1--v6): treino do zero em escala reduzida.** Esta trilha implementa e itera um pipeline completo (curadoria, limpeza, tokenização e treinamento) para quantificar, na prática, o custo e a sensibilidade de treinar um modelo Transformer do zero. O objetivo não é competir com modelos de fronteira, mas evidenciar a dificuldade inerente (dados, engenharia e tempo de GPU) e registrar as decisões que levaram à versão final (v6).
    \item **Linha viável: adaptação via LoRA/QLoRA.** A partir de um modelo open-source de maior capacidade, aplica-se pós-treinamento eficiente em GPU para maximizar utilidade prática dentro do mesmo orçamento computacional. No escopo mínimo viável executado neste trabalho, esta trilha contempla (i) *continued pretraining* (CPT) com amostras do BrWaC para adaptação linguística e (ii) *supervised fine-tuning* (SFT) com um dataset instrucional em português brasileiro (Canarim), seguido de avaliação extrínseca em QA, sumarização e reescrita.


O objetivo deste projeto é, portanto, entregar (i) um protótipo funcional com evidências quantitativas do custo de uma linha ``do zero'' (v1--v6) e (ii) evidência empírica de uma alternativa pragmática de pós-treinamento eficiente (CPT/SFT via LoRA/QLoRA) em GPU única, com logs, métricas e artefatos reprodutíveis. Os próximos capítulos apresentam o referencial teórico que embasa estas escolhas, a descrição da metodologia aplicada e a análise dos resultados obtidos.

## Contribuições

    \item **Pipeline reprodutível e rastreável (v1--v6):** implementação iterativa de um fluxo completo (dados, tokenização, treino, avaliação) para explicitar custos e desafios do treinamento do zero em escala reduzida.
    \item **Evidência do impacto da tokenização:** registro sistemático de métricas intrínsecas, amostras padronizadas e *guardrails* para mostrar que decisões de tokenização podem dominar a qualidade percebida.
    \item **Execução de pós-treinamento eficiente (Trilha 2):** CPT e SFT via QLoRA em um modelo 8B open-source, com logs estruturados e custos reportados.
    \item **Avaliação extrínseca comparável:** protocolo de QA/sumarização/reescrita com repetição em múltiplas seeds de amostragem e artefatos versionados para auditoria.


## Primeiras tentativas de processamento de linguagem natural

O interesse em fazer máquinas compreenderem e gerarem linguagem humana remonta aos primórdios da computação. Uma das iniciativas mais emblemáticas desse período foi o desenvolvimento do programa **ELIZA**, criado por Joseph Weizenbaum no MIT, durante a década de 1960 [weizenbaum1966]. ELIZA foi projetada para simular um psicoterapeuta rogeriano, interagindo com usuários por meio de diálogos escritos. Seu funcionamento baseava-se em regras simples de reconhecimento de padrões e substituição de palavras-chave, permitindo que o sistema identificasse termos relevantes nas frases do usuário e respondesse com perguntas ou afirmações genéricas.

A principal característica de ELIZA era sua abordagem simbólica e determinística: não havia aprendizado automático, mas sim um conjunto fixo de scripts e regras pré-definidas [jurafsky2023]. Apesar dessa simplicidade, ELIZA surpreendeu ao criar a ilusão de compreensão, levando muitos usuários a atribuírem ao programa uma inteligência que ele não possuía de fato. Esse fenômeno ficou conhecido como o "efeito ELIZA", ilustrando como interações superficiais podem ser percebidas como inteligentes quando envolvem linguagem natural [weizenbaum1966; jurafsky2023].

Além de ELIZA, outros sistemas simbólicos surgiram nas décadas seguintes, como parsers sintáticos e chatbots baseados em regras. Um exemplo notável é o sistema SHRDLU, desenvolvido por Terry Winograd [winograd1972], que utilizava gramáticas formais e árvores sintáticas para decompor frases e responder a comandos em um ambiente restrito. Esses sistemas buscavam decompor frases em estruturas gramaticais ou responder a comandos específicos, utilizando gramáticas formais e árvores sintáticas [jurafsky2023]. Embora tenham representado avanços importantes para a época, esses métodos enfrentavam limitações severas: eram altamente dependentes de regras manuais, pouco escaláveis e incapazes de lidar com a ambiguidade, a variabilidade e a riqueza semântica da linguagem humana.

Os principais problemas encontrados nessas abordagens estavam relacionados à rigidez dos sistemas e à incapacidade de generalizar para contextos não previstos pelos desenvolvedores. A ausência de mecanismos de aprendizado impedia que os sistemas evoluíssem a partir de novas interações ou dados, restringindo seu uso a domínios muito específicos e controlados [jurafsky2023].

Apesar dessas limitações, as primeiras tentativas de processamento de linguagem natural foram fundamentais para estabelecer as bases conceituais do campo [turing1950; jurafsky2023]. A rigidez e a falta de escalabilidade dos sistemas simbólicos deixaram claro que uma nova abordagem era necessária — uma que pudesse aprender com dados, em vez de ser explicitamente programada. Esse foi o vácuo que o aprendizado de máquina começou a preencher.


## Evolução para Machine Learning

Com o avanço da computação e o aumento da disponibilidade de dados digitais, as limitações dos sistemas puramente simbólicos tornaram-se cada vez mais evidentes. Isso impulsionou a transição para abordagens baseadas em **aprendizado de máquina (Machine Learning, ML)**, especialmente a partir das décadas de 1980 e 1990 [sebastiani2002; jurafsky2023]. O objetivo passou a ser permitir que os sistemas aprendessem padrões linguísticos a partir de exemplos, em vez de depender exclusivamente de regras manuais.

Os primeiros algoritmos de ML aplicados ao processamento de linguagem natural incluíam métodos estatísticos como **Naive Bayes**, **máquinas de vetores de suporte (SVM)** [joachims1998] e **árvores de decisão**. Essas técnicas eram empregadas em tarefas como classificação de textos, análise de sentimentos, detecção de spam e extração de informações [sebastiani2002]. Uma característica marcante desse período foi o uso de representações vetoriais simples, como o modelo **bag-of-words** e o **TF-IDF** (Term Frequency-Inverse Document Frequency), que transformavam textos em conjuntos de números baseados na frequência das palavras [salton1988].

Essas abordagens trouxeram avanços significativos em relação aos sistemas simbólicos. Ao permitir que modelos fossem treinados com grandes volumes de dados rotulados, tornou-se possível capturar regularidades estatísticas da linguagem e adaptar os sistemas a diferentes domínios e tarefas [sebastiani2002; jurafsky2023]. Além disso, a automação do processo de aprendizado reduziu a dependência de especialistas em linguística para a criação de regras, tornando o desenvolvimento de aplicações de processamento de linguagem natural mais ágil e escalável.

No entanto, os métodos baseados em machine learning tradicional também apresentavam limitações importantes. Por tratarem o texto como uma coleção de palavras independentes, ignoravam a ordem e o contexto em que as palavras apareciam, o que dificultava a compreensão de estruturas sintáticas e semânticas mais complexas [jurafsky2023]. Além disso, a necessidade de grandes conjuntos de dados rotulados para treinamento era um desafio, especialmente em domínios especializados ou em idiomas com menos recursos.

Apesar desses obstáculos, a adoção do aprendizado de máquina representou um avanço fundamental para o campo. Ela abriu caminho para o desenvolvimento de técnicas mais sofisticadas, capazes de capturar relações contextuais e semânticas, e preparou o terreno para a próxima grande revolução: o **aprendizado profundo (Deep Learning)** [jurafsky2023].


## Avanço com Deep Learning

A partir do final dos anos 2000, o campo do processamento de linguagem natural (PLN) foi profundamente transformado pelo **aprendizado profundo (Deep Learning)**. Esse avanço foi impulsionado pelo aumento do poder computacional, pela disponibilidade de grandes volumes de dados e pelo desenvolvimento de novas arquiteturas de redes neurais [lecun2015; jurafsky2023].

O primeiro grande desafio que o deep learning superou foi a representação de palavras. Modelos anteriores, como bag-of-words, tratavam as palavras como itens isolados. A revolução veio com os **word embeddings** a partir de 2013. Modelos como **Word2Vec** [mikolov2013] e **GloVe** [pennington2014] aprenderam a representar palavras como vetores densos em um espaço multidimensional. Nessa representação, palavras com significados semelhantes ficavam próximas, permitindo que os modelos capturassem relações semânticas e sintáticas (por exemplo, a relação entre "rei" e "rainha" ser similar à de "homem" e "mulher"). Essa capacidade de representar o significado das palavras foi um pilar para os avanços subsequentes.

Com palavras representadas de forma significativa, o desafio seguinte foi compreender a ordem e o contexto em que elas aparecem. As **Redes Neurais Recorrentes (RNNs)** foram a primeira arquitetura de deep learning projetada para processar sequências, mantendo um estado interno ou "memória" do que foi visto anteriormente. No entanto, as RNNs tradicionais sofriam com o problema do gradiente desvanecente, o que as impedia de capturar dependências em trechos longos de texto [bengio1994]. Para solucionar essa limitação, foram desenvolvidas variantes mais robustas, como as **Long Short-Term Memory (LSTM)** [hochreiter1997] e as **Gated Recurrent Units (GRU)** [cho2014]. Essas arquiteturas introduziram "portões" (gates) que controlam o fluxo de informação, permitindo que a rede se lembre de informações relevantes por períodos muito mais longos.

Paralelamente, as **Redes Neurais Convolucionais (CNNs)**, tradicionalmente usadas em visão computacional, foram adaptadas com sucesso para o PLN [kim2014]. Em vez de processar o texto sequencialmente, as CNNs aplicam filtros que deslizam sobre as sequências de palavras para identificar padrões locais, como frases ou expressões idiomáticas, independentemente de sua posição no texto.

A combinação de word embeddings com arquiteturas sequenciais (LSTMs, GRUs) e convolucionais (CNNs) permitiu que os modelos de linguagem alcançassem um novo patamar de compreensão contextual, preparando o terreno para a arquitetura que viria a definir a era moderna do PLN: os Transformers.

## A Revolução dos Transformers e os Modelos de Linguagem de Grande Escala (LLMs)

Em 2017, o campo do processamento de linguagem natural foi novamente transformado com a publicação do artigo “Attention is All You Need” [vaswani2017attention], que apresentou ao mundo a arquitetura **Transformer**. Esse modelo representou uma ruptura fundamental com as abordagens sequenciais, como as RNNs. A principal inovação dos Transformers é a capacidade de processar todas as palavras de uma sequência simultaneamente, permitindo uma paralelização massiva e um treinamento muito mais eficiente.

O coração do Transformer é o **mecanismo de atenção** (*attention mechanism*). Em vez de depender de uma memória sequencial, a atenção permite que o modelo, ao analisar uma palavra, pondere dinamicamente a importância de todas as outras palavras na sequência. Matematicamente, para cada palavra, o modelo gera vetores de *query* (consulta), *key* (chave) e *value* (valor). A pontuação de atenção é calculada comparando o *query* de uma palavra com a *key* de todas as outras, determinando o quanto de "foco" cada palavra deve receber. O resultado é uma representação contextual rica, que captura relações complexas de curto e longo alcance, resolvendo ambiguidades que eram desafiadoras para modelos anteriores.

A eficiência e o poder de contextualização dos Transformers abriram caminho para a era dos **Modelos de Linguagem de Grande Escala (Large Language Models, LLMs)**. A nova paradigma consistia em duas fases: primeiro, o **pré-treinamento**, no qual um modelo é treinado em uma quantidade massiva de texto não rotulado para aprender padrões gerais da linguagem, gramática, fatos e raciocínio. Em seguida, a fase de **ajuste fino** (*fine-tuning*), onde o modelo pré-treinado é adaptado para tarefas específicas (como classificação, tradução ou resposta a perguntas) com um conjunto de dados muito menor e rotulado.

Essa abordagem levou a avanços sem precedentes, com o surgimento de modelos icônicos:

    \item **BERT (Bidirectional Encoder Representations from Transformers, 2018):** Desenvolvido pelo Google, o BERT inovou ao utilizar o encoder do Transformer para aprender representações de palavras considerando o contexto de ambos os lados (esquerdo e direito) simultaneamente, alcançando resultados estado da arte em uma vasta gama de tarefas de compreensão de linguagem [devlin2018].
    \item **GPT (Generative Pre-trained Transformer, 2018 em diante):** Criado pela OpenAI, a família de modelos GPT utiliza o decoder do Transformer com foco na geração de texto. Do GPT-1 ao GPT-3 e além, esses modelos demonstraram uma capacidade impressionante de gerar textos coerentes, criativos e contextualmente relevantes a partir de um prompt [radford2018; brown2020].
    \item **T5 (Text-to-Text Transfer Transformer, 2020):** Também do Google, o T5 propôs um framework unificado, tratando todas as tarefas de PLN como um problema de "texto-para-texto". O modelo recebe uma instrução em texto (ex: "traduza Inglês para Português: ...") e gera a saída também em texto, simplificando a aplicação do modelo a diferentes problemas [raffel2020].

A trajetória desses modelos, especialmente da série GPT, demonstrou que o escalonamento massivo — aumentando exponencialmente a quantidade de parâmetros e os dados de treinamento — resultava em capacidades emergentes e um salto de performance. O avanço, contudo, não se limitou a essa corrida por escala. À medida que os LLMs se tornaram mais capazes, o desafio de alinhar seu comportamento a objetivos e valores humanos tornou-se crítico. Surgiram, então, técnicas de pós-treinamento para refinar as respostas dos modelos, como o Aprendizado por Reforço com Feedback Humano (RLHF), que utiliza avaliações humanas para guiar o aprendizado e tornar os modelos mais úteis e seguros [bai2023alignment].

### Pós-treinamento: adaptação e alinhamento


Embora o termo *fine-tuning* seja frequentemente usado de forma ampla, é útil diferenciar objetivos distintos na fase de pós-treinamento: (i) adaptação linguística/domínio, (ii) seguimento de instruções e (iii) alinhamento por preferências. No primeiro caso, o *continued pretraining* (CPT) continua o pré-treinamento com o mesmo objetivo auto-supervisionado (predição do próximo token), porém com dados do idioma/domínio de interesse [gururangan2020dont]. No segundo, o *supervised fine-tuning* (SFT) — frequentemente referido como *instruction tuning* quando usa pares instrução--resposta — ensina o modelo a responder a comandos em formatos controlados (e.g., QA, sumarização, reescrita), com exemplos rotulados [ouyang2022training]. Por fim, técnicas de alinhamento como RLHF refinam o comportamento a partir de sinais de preferência humana, complementando o SFT ao penalizar respostas indesejadas e reforçar as preferidas [bai2023alignment].

Do ponto de vista computacional, essas etapas tornaram-se viáveis em hardware de consumo com técnicas de *parameter-efficient fine-tuning* (PEFT) --- como LoRA --- nas quais o modelo base é mantido congelado e apenas adaptadores pequenos são treinados, representando uma fração dos parâmetros totais [hu2021lora]. QLoRA estende essa estratégia ao manter o modelo base quantizado (e.g., 4-bit) enquanto treina os adaptadores LoRA em maior precisão, reduzindo VRAM e viabilizando o ajuste de modelos maiores em GPU única [dettmers2023qlora]. Estes conceitos são detalhados no Capítulo~ e aplicados na trilha de pós-treinamento descrita no Capítulo~.

Paralelamente, o ecossistema de IA dividiu-se. Enquanto modelos de ponta como os da série GPT permaneceram proprietários, um movimento de código aberto (open-source) ganhou força, buscando democratizar o acesso a essa tecnologia. Iniciativas como LLaMA [touvron2023llama] e Qwen [qwen2024team] disponibilizaram modelos poderosos para a comunidade, viabilizando a pesquisa e o desenvolvimento de aplicações adaptadas a contextos específicos — exatamente a abordagem adotada neste trabalho.

## A Era dos LLMs e o Desafio da Representação Cultural

A ascensão dos modelos de linguagem de grande escala (LLMs), como BERT, GPT e T5, representa o estado da arte em processamento de linguagem natural. Seu sucesso é inegável, mas também expôs uma limitação fundamental: a sua perspectiva predominantemente anglocêntrica.

Quando aplicados a outros idiomas, como o português do Brasil, esses modelos podem apresentar um desempenho satisfatório em tarefas genéricas, mas frequentemente falham em capturar as complexidades e especificidades locais. Gírias, expressões idiomáticas, contextos culturais, referências regionais e a própria estrutura semântica da língua podem ser mal interpretados ou simplesmente ignorados por modelos que não foram treinados nativamente com dados brasileiros [caswell2021].

Essa lacuna cria uma barreira para o desenvolvimento de aplicações de IA verdadeiramente eficazes e culturalmente conscientes para o contexto brasileiro. Mais do que uma simples questão de tradução, trata-se de um desafio de representação. A dependência de modelos estrangeiros pode perpetuar vieses e criar tecnologias que não atendem adequadamente às necessidades da população local.

É nesse cenário que surge a necessidade crítica de desenvolver LLMs 100\% brasileiros. Um modelo treinado desde o início com um corpus massivo e diversificado de textos em português do Brasil tem o potencial de não apenas compreender melhor as sutilezas do idioma, mas também de incorporar o vasto conhecimento cultural, social e histórico do país. Este trabalho se insere exatamente nessa fronteira, buscando contribuir para a construção de uma inteligência artificial que fale, de fato, a nossa língua.


Este capítulo consolida os **fundamentos técnicos essenciais** para entender as escolhas de arquitetura, as limitações de hardware e a metodologia experimental do Capítulo~. O foco é didático: apresentar apenas o necessário para acompanhar (i) a Trilha 1 (v1--v6) e (ii) a Trilha 2 (CPT/SFT via QLoRA), evitando aprofundamentos que não foram implementados.

\begin{figure}[ht]

\fbox{\begin{minipage}{0.95\textwidth}
\small
\begin{tabularx}{\linewidth}{@{}p{3.2cm}X@{}}
**Seção 3.1** & Tokenização (char-level $\rightarrow$ subword) $\rightarrow$ usada na Trilha 1 (v1--v6); a escolha do tokenizer se mostrou determinante para legibilidade/qualidade. \\
**Seção 3.2** & Arquiteturas na Trilha 1 (RNN $\rightarrow$ Transformer) $\rightarrow$ v1--v4 (modelos sequenciais) e v5--v6 (Transformer em GPU única). \\
**Seção 3.3** & Compute e VRAM (GPU única) $\rightarrow$ justifica escala reduzida em v1--v6 e limitações de contexto/treino. \\
**Seção 3.4** & LoRA/QLoRA $\rightarrow$ viabiliza a Trilha 2 (CPT e SFT) em 1x RTX 4090 (24 GB). \\
**Seção 3.5** & Scaling laws (Kaplan/Chinchilla) $\rightarrow$ contextualiza por que v1--v6 não compete com modelos de fronteira e por que pós-treinamento é o caminho prático. \\
\end{tabularx}
\end{minipage}}
{Roadmap de conexão entre fundamentos (Cap. 3) e experimentos (Cap. 4).}

\end{figure}

## Tokenização (v1--v6): char-level e subword

Um modelo de linguagem não consome texto diretamente: ele opera sobre uma sequência discreta de **tokens** (IDs), definida por um **tokenizer** e seu vocabulário. Essa escolha afeta simultaneamente (i) a **qualidade** (fragmentação, legibilidade e consistência lexical) e (ii) o **custo** (comprimento efetivo de sequência e taxa de tokens/segundo).

Para fixar a terminologia usada ao longo do capítulo: neste trabalho, **token** é a unidade básica que o modelo lê e prediz (por exemplo, um caractere, uma subpalavra ou um pedaço de palavra). Já o **tokenizer** é o componente (procedimento + vocabulário) que transforma texto em tokens/IDs (*encode*) e permite reconstruir texto a partir de IDs (*decode*).

### Char-level

Em tokenização *char-level*, cada caractere é um token. É uma escolha simples, sem OOV (*out-of-vocabulary*) e sem necessidade de treinar um tokenizer, mas tende a produzir sequências muito mais longas (mais passos de predição), o que aumenta o custo computacional e pode dificultar o aprendizado de regularidades semânticas de alto nível.

\noindent**Exemplo (ilustrativo).** Texto `casa` $\rightarrow$ tokens `[c, a, s, a]` (4 tokens).

### Subword (SentencePiece)

Em tokenização *subword*, palavras são segmentadas em unidades menores (subpalavras) aprendidas a partir de um corpus (por exemplo, SentencePiece). SentencePiece é uma biblioteca amplamente usada para treinar tokenizers *subword* diretamente sobre texto cru, com algoritmos como *unigram* e BPE. Em geral, isso reduz o comprimento da sequência em relação ao char-level e melhora a modelagem de morfologia e composição de palavras, especialmente em português. Em contrapartida, um tokenizer mal ajustado pode gerar **fragmentação excessiva** (muitas peças curtas), o que degrada a legibilidade mesmo quando métricas intrínsecas (loss/perplexidade) parecem aceitáveis.

\noindent**Exemplo (ilustrativo).** Texto `infraestrutura` $\rightarrow$ tokens `[infra, estrutura]` ou `[in, fra, estrutura]`, dependendo do vocabulário aprendido.

No Capítulo~, as versões v1--v3 utilizam abordagem char-level para viabilizar iteração rápida e eliminar variáveis adicionais no início do pipeline. A v4 introduz subwords (SentencePiece), representando um salto qualitativo perceptível nos exemplos. Nas versões seguintes (v5--v6), a principal descoberta empírica foi que **a escolha do tokenizer pode dominar ganhos arquiteturais marginais**, motivando a inclusão de *guardrails* e análises sistemáticas de amostras.

## Arquiteturas na Trilha 1 (v1--v6): RNN \texorpdfstring{$\rightarrow${->} Transformer}

### Modelos sequenciais (v1--v4): RNN/GRU/LSTM

Nas primeiras versões, a Trilha 1 utilizou modelos sequenciais (RNNs e variações como GRU/LSTM). Esses modelos processam a sequência token a token, mantendo um *estado oculto* que resume o contexto anterior. A escolha foi adequada para prototipagem: implementação simples, treinamento viável em GPU única e foco na construção do pipeline (dados, validação e geração).

\noindent**Exemplo.** Dada uma sequência de tokens $[t_1,t_2,t_3]$, uma RNN atualiza um estado $h_t$ passo a passo (primeiro $t_1$, depois $t_2$, etc.) e usa $h_t$ para predizer o próximo token. Isso contrasta com Transformers, que calculam atenção entre posições e exploram paralelismo (discutido a seguir).

### Transformer (v5--v6): atenção e paralelização

Transformers [vaswani2017attention] substituem recorrência por **atenção** para modelar dependências entre tokens. A ideia central é que, ao processar uma posição $i$, o modelo calcula pesos que indicam o quanto cada outra posição $j$ deve contribuir para a representação final daquela posição.

Em sua forma padrão (*scaled dot-product attention*), para matrizes de consultas $Q$, chaves $K$ e valores $V$, define-se:
\begin{equation}
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\end{equation}

O *multi-head attention* replica esse mecanismo em múltiplas ``cabeças'', permitindo que diferentes padrões de relacionamento (por exemplo, concordância, contexto temático, dependências longas) sejam capturados em paralelo, e depois combinados.

No Capítulo~, a Trilha 1 usa v1--v4 com modelos sequenciais (RNN/GRU/LSTM) para viabilizar prototipagem e evolução rápida do pipeline. A partir de v5, a adoção de Transformer permite explorar paralelização em GPU, mas introduz custo quadrático no comprimento de sequência, o que impõe limites práticos de `seq\_len` sob VRAM fixa (discutido a seguir).

## Requisitos computacionais em GPU única: compute e VRAM

Em treinamento de modelos autoregressivos, o custo computacional cresce com (i) o tamanho do modelo (parâmetros) e (ii) a quantidade de dados (tokens). Além disso, a **VRAM** é frequentemente o gargalo principal em GPU única, pois precisa acomodar pesos, estados do otimizador e ativações.

### Regra prática de compute (ordem de grandeza)

Como aproximação útil, o compute total de treino pode ser escrito como:
\begin{equation}
C \approx 6ND
\end{equation}
onde $N$ é o número de parâmetros e $D$ o número de tokens vistos no treinamento (contexto de *scaling laws* [kaplan2020scaling; hoffmann2022training]). O objetivo aqui é didático: mostrar por que treinar do zero em escala de LLMs rapidamente sai do orçamento de uma única GPU.

### Por que 7B não cabe em 24 GB (treino completo)

Na prática, o treinamento é feito com otimizadores adaptativos como o Adam, que mantêm estados adicionais por parâmetro (estimativas do primeiro e do segundo momento dos gradientes). Em configurações típicas, esses estados ficam em 32 bits (*precisão plena*), mesmo quando o treinamento usa precisão mista para acelerar. Assim, em ordem de grandeza, cada parâmetro requer: parâmetro, gradiente e dois momentos, o que leva a uma conta de VRAM que excede 24 GB para modelos de bilhões de parâmetros, motivando (i) protótipos reduzidos na Trilha 1 e (ii) pós-treinamento eficiente na Trilha 2.

No Capítulo~, essa restrição aparece diretamente: (i) v1--v6 são modelos pequenos para permitir iteração e rastreabilidade em GPU única; (ii) a Trilha 2 usa QLoRA para adaptar um modelo maior (LLaMA~3.1-8B) com custo e VRAM viáveis.

## Pós-treinamento eficiente: LoRA e QLoRA

O pós-treinamento eficiente busca adaptar modelos grandes com poucos parâmetros treináveis. Em **LoRA** [hu2021lora], em vez de atualizar uma matriz de pesos $W$, aprende-se uma atualização de *rank* reduzido (*low-rank*):
\begin{equation}
W' = W + BA
\end{equation}
onde $A\in\mathbb{R}^{r\times k}$ e $B\in\mathbb{R}^{d\times r}$ usam uma dimensão interna $r$ pequena (por exemplo, 8 ou 16). Em vez de treinar todos os $d\cdot k$ elementos de $W\in\mathbb{R}^{d\times k}$, treina-se apenas $r(d+k)$ parâmetros nos adaptadores: quanto maior $r$, maior a capacidade do ajuste, mas também maior o custo em VRAM e tempo. Nesse sentido, $\Delta W=BA$ fica restrita a um subespaço de dimensão $r$.

Em termos de ordem de grandeza, a redução pode ser grande: por exemplo, com $d=k=4096$ e $r=8$, atualizar $W$ exigiria treinar $\approx 16{,}8$ milhões de parâmetros, enquanto LoRA treina $r(d+k)=65\,536$ parâmetros (redução de $\sim 256\times$). Na prática, a economia de VRAM vem do fato de que gradientes e estados do otimizador (por exemplo, do Adam) precisam ser mantidos apenas para os parâmetros treináveis; ao treinar apenas os adaptadores, evita-se armazenar estados completos para todos os pesos do modelo base.

**QLoRA** [dettmers2023qlora] combina LoRA com quantização do modelo base (por exemplo, 4-bit), reduzindo o consumo de VRAM e tornando possível treinar/adaptar em GPU única com 24 GB, sem re-treinar todos os pesos.

No Capítulo~, a Trilha 2 usa QLoRA para executar (i) CPT em PT-BR (BrWaC 10k) e (ii) SFT instrucional (Canarim 10k), com avaliação extrínseca em tarefas downstream. O foco é maximizar utilidade prática dentro de um orçamento computacional realista.

## Scaling laws: implicações para a estratégia em duas trilhas

*Scaling laws* são regularidades **empíricas** observadas em grandes famílias de experimentos: elas descrevem como métricas como *loss* tendem a diminuir quando se aumenta (i) o tamanho do modelo (número de parâmetros), (ii) a quantidade de dados (tokens) e (iii) o compute total. Em geral, essas curvas seguem leis de potência com **retornos decrescentes** [kaplan2020scaling], e por isso são úteis para estimar ordem de grandeza de custo e ganhos esperados. Hoffmann et al. [hoffmann2022training] reforçam que, dado um budget de compute, há um balanço ótimo entre tamanho do modelo e quantidade de dados (regra conhecida como *compute-optimal*).

Essas leis motivam a estrutura do projeto apresentada no Capítulo~: v1--v6 não buscam competir com modelos de fronteira; seu valor está em evidenciar custos, armadilhas e decisões de pipeline (limpeza/tokenização/treino) em um cenário reprodutível. A Trilha 2, por sua vez, opera em um regime de escala mais próximo do uso prático ao adaptar um modelo 8B já pré-treinado, demonstrando ganhos mensuráveis em tarefas de QA, sumarização e reescrita com baixo custo.


## Estratégia experimental

O trabalho prático foi estruturado para sustentar a narrativa central desta monografia: (i) quantificar, em uma escala reduzida e reprodutível, a dificuldade de treinar um modelo do zero (v1--v6) e (ii) contrastar esta abordagem com uma alternativa viável sob restrições reais de hardware, baseada em pós-treinamento eficiente (LoRA/QLoRA) de um modelo open-source.

**Trilha 1 -- Treinamento do zero (v1--v6).** Implementa-se e itera um pipeline completo de dados (limpeza e tokenização) e treinamento de um Transformer em português (a partir do BrWaC), registrando decisões e métricas por versão. O objetivo é fornecer evidência concreta do custo (tempo, engenharia e dados) mesmo em uma escala muito menor do que modelos de fronteira.

**Trilha 2 -- Pós-treinamento eficiente (LoRA/QLoRA).** Utiliza-se um modelo base open-source de maior capacidade, realizando (i) *continued pretraining* (CPT) em português e (ii) *supervised fine-tuning* (SFT) com dados instrucionais em PT-BR. Esta trilha busca maximizar utilidade prática no mesmo orçamento computacional, e é detalhada junto aos experimentos correspondentes.

Para garantir rastreabilidade, os artefatos experimentais (tokenizers, modelos, mapeamentos, logs e amostras) são armazenados no repositório do projeto (diretório [versions](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions)).

## Análise Técnica do Ambiente Computacional

O desenvolvimento de LLMs requer uma análise rigorosa dos recursos computacionais disponíveis para determinar estratégias viáveis de implementação. Esta seção descreve o ambiente computacional utilizado e as principais restrições que motivam as escolhas metodológicas.

### Ambiente de execução (Vast.ai)

Os experimentos desta monografia foram executados em instâncias sob demanda na Vast.ai, utilizando GPUs de consumo que variaram ao longo das versões: v1--v4 em RTX 3080, v5/v5b em RTX 3090 (24 GB) e v6 e Trilha 2 em RTX 4090 (24 GB). Para garantir reprodutibilidade, logs e artefatos (tokenizers, modelos, mapeamentos, histórico de treino e amostras) foram armazenados no repositório do projeto, com os resultados consolidados do v6 no diretório [versions/v6-release-candidate](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/v6-release-candidate) e os artefatos da Trilha 2 em [versions/trilha2-lora](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/trilha2-lora).

### Análise de Capacidades para Treinamento de LLMs

#### Limitações Críticas de Memória GPU

Para treinamento de LLMs, a VRAM disponível é o fator limitante principal. Baseado nas análises dos fundamentos técnicos, os requisitos de memória para diferentes tamanhos de modelo são:

**Modelo de 7B parâmetros (treinamento completo):**
\begin{align}
\text{Memória}_{\text{parâmetros}} &= 7 \times 10^9 \times 4 \text{ bytes} = 28 \text{ GB} \\
\text{Memória}_{\text{gradientes}} &= 7 \times 10^9 \times 4 \text{ bytes} = 28 \text{ GB} \\
\text{Memória}_{\text{otimizador}} &= 7 \times 10^9 \times 8 \text{ bytes} = 56 \text{ GB} \\
\text{Memória}_{\text{total}} &= 28 + 28 + 56 = 112 \text{ GB}
\end{align}

**Análise de viabilidade:**
\begin{equation}
\text{Déficit}_{\text{VRAM}} = 112 \text{ GB} - 24 \text{ GB} = 88 \text{ GB}
\end{equation}

O déficit de 88 GB evidencia que o treinamento completo de um modelo de 7B parâmetros é inviável em uma única GPU de 24 GB sem estratégias avançadas de paralelismo e otimizações de memória. Por esse motivo, a trilha experimental (v1--v6) foi conduzida em escala reduzida, e a trilha prática prioriza pós-treinamento eficiente (LoRA/QLoRA) em modelos pré-treinados.

#### Custo computacional

Além da VRAM, o custo de treinamento cresce com o número de parâmetros e com o número de tokens processados. Mesmo protótipos reduzidos podem demandar muitas horas de GPU; por isso, este trabalho reporta tempos observados e logs reprodutíveis, e contrasta treino do zero (v1--v6) com alternativas de pós-treinamento eficientes em GPU única.

Para sustentar a evidência dos custos reportados, a Tabela~ resume as fontes (no repositório) usadas para computar tempos e verificar a GPU utilizada em cada execução principal (valores arredondados). Os tempos apresentados nas Tabelas~,  e  foram extraídos desses logs.

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.0cm}|c|X|X|}
\hline
**Execução** & **Tempo (h)** & **Fonte (tempo)** & **Fonte (GPU)** \\
\hline
`v1` & 1.1 & [versions/v1-char-rnn/logs/train_brwac_v1_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v1-char-rnn/logs/train_brwac_v1_20k.json) & [versions/v1-char-rnn/logs/train_brwac_v1_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v1-char-rnn/logs/train_brwac_v1_20k.json) (RTX 3080) \\
`v2` & 0.9 & [versions/v2-char-lm/logs/train_brwac_v2_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v2-char-lm/logs/train_brwac_v2_20k.json) & [versions/v2-char-lm/logs/train_brwac_v2_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v2-char-lm/logs/train_brwac_v2_20k.json) (RTX 3080) \\
`v3` & 2.0 & [versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json) & [versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json) (RTX 3080) \\
`v4_subword` & 1.2 & [versions/v4-subword-lstm/logs/train_brwac_v4_subword.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword.json) & [versions/v4-subword-lstm/logs/train_brwac_v4_subword.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword.json) (RTX 3080) \\
`v4_subword_ep6` & 3.7 & [versions/v4-subword-lstm/logs/train_brwac_v4_subword_ep6.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword_ep6.json) & [versions/v4-subword-lstm/logs/train_brwac_v4_subword_ep6.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword_ep6.json) (RTX 3080) \\
`v4k_bf_ep2` & 1.2 & [versions/v4-subword-lstm/logs/train_brwac_v4_subword_v4k_bf_ep2.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword_v4k_bf_ep2.json) & [versions/v4-subword-lstm/logs/train_brwac_v4_subword_v4k_bf_ep2.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_subword_v4k_bf_ep2.json) (RTX 3080) \\
`v4k_bf_lstm_ep2` & 1.4 & [versions/v4-subword-lstm/logs/train_brwac_v4_lstm_v4k_bf_ep2.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_lstm_v4k_bf_ep2.json) & [versions/v4-subword-lstm/logs/train_brwac_v4_lstm_v4k_bf_ep2.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v4-subword-lstm/logs/train_brwac_v4_lstm_v4k_bf_ep2.json) (RTX 3080) \\
`v5_main` & 3.2 & [versions/v5-transformer/logs/train_v5_main.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v5-transformer/logs/train_v5_main.json) & [versions/v5-transformer/logs/train_v5_scale.log](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v5-transformer/logs/train_v5_scale.log) (RTX 3090) \\
`v5b_long` & 2.9 & [versions/v5b-transformer/logs/metrics_v5b_long](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/v5b-transformer/logs/metrics_v5b_long) (`metrics_epoch_*.json`) & [versions/v5b-transformer/logs/long_stdout_20251105_relaunch.log](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v5b-transformer/logs/long_stdout_20251105_relaunch.log) (RTX 3090) \\
`v6_rc16k_v5b` & 13.3 & [versions/v6-release-candidate/logs/train_v6_rc16k_v5b.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v6-release-candidate/logs/train_v6_rc16k_v5b.json) & [versions/v6-release-candidate/logs/run_v6_rc16k_v5b_head.txt](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v6-release-candidate/logs/run_v6_rc16k_v5b_head.txt) (RTX 4090) \\
`v6_rc16k_v5b_aligned` & 16.6 & [versions/v6-release-candidate/logs/train_v6_rc16k_v5b_aligned.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v6-release-candidate/logs/train_v6_rc16k_v5b_aligned.json) & [versions/v6-release-candidate/logs/train_v6_rc16k_v5b_aligned_head.txt](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v6-release-candidate/logs/train_v6_rc16k_v5b_aligned_head.txt) (RTX 4090) \\
`cpt_qlora_llama31_8b_brwac10k` & 2.35 & [versions/trilha2-lora/logs/train_cpt_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_cpt_qlora.json) & [versions/trilha2-lora/logs/train_cpt_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_cpt_qlora.json) (RTX 4090) \\
`sft_qlora_llama31_8b_instruct_canarim10k` & 1.06 & [versions/trilha2-lora/logs/train_sft_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_sft_qlora.json) & [versions/trilha2-lora/logs/train_sft_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_sft_qlora.json) (RTX 4090) \\
`eval_trilha2_extrinsic_3seeds` & 1.81 & [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json) & [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json) (RTX 4090) \\
\hline
\end{tabularx}
{Rastreabilidade de tempo e hardware por execução (logs no repositório). Em alguns casos (e.g., `v5\_main`), tempo e hardware são registrados em arquivos distintos.}

\end{table}

Para materializar o componente econômico do ``orçamento computacional'', a Tabela~ apresenta preços de referência por hora (Vast.ai, percentil 25) utilizados para estimar o custo por execução em GPU única. Esses valores servem como aproximação (os preços variam no tempo e não incluem custos indiretos como armazenamento e tempo ocioso).

\begin{table}[ht]

\begin{tabular}{|l|c|}
\hline
**GPU (1x)** & **Preço (USD/h, P25)** \\
\hline
RTX 3080 & \$0.07 \\
RTX 3090 & \$0.13 \\
RTX 4090 & \$0.30 \\
\hline
\end{tabular}
{Preços de referência por hora (Vast.ai, P25) usados nas estimativas de custo.}

\end{table}

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}X|>{\arraybackslash}p{1.6cm}|>{\arraybackslash}p{1.2cm}|>{\arraybackslash}p{1.8cm}|>{\arraybackslash}p{1.8cm}|}
\hline
**Execução** & **GPU** & **Tempo (h)** & **Preço (USD/h)** & **Custo est. (USD)** \\
\hline
`v1` & RTX 3080 & 1.1 & 0.07 & 0.08 \\
`v2` & RTX 3080 & 0.9 & 0.07 & 0.06 \\
`v3` & RTX 3080 & 2.0 & 0.07 & 0.14 \\
`v4_subword` & RTX 3080 & 1.2 & 0.07 & 0.08 \\
`v4_subword_ep6` & RTX 3080 & 3.7 & 0.07 & 0.26 \\
`v4k_bf_ep2` & RTX 3080 & 1.2 & 0.07 & 0.08 \\
`v4k_bf_lstm_ep2` & RTX 3080 & 1.4 & 0.07 & 0.10 \\
`v5_main` & RTX 3090 & 3.2 & 0.13 & 0.42 \\
`v5b_long` & RTX 3090 & 2.9 & 0.13 & 0.38 \\
`v6_rc16k_v5b` & RTX 4090 & 13.3 & 0.30 & 3.99 \\
`v6_rc16k_v5b_aligned` & RTX 4090 & 16.6 & 0.30 & 4.98 \\
`cpt_qlora_llama31_8b_brwac10k` & RTX 4090 & 2.35 & 0.30 & 0.71 \\
`sft_qlora_llama31_8b_instruct_canarim10k` & RTX 4090 & 1.06 & 0.30 & 0.32 \\
`eval_trilha2_extrinsic_3seeds` & RTX 4090 & 1.81 & 0.30 & 0.54 \\
\hline
\end{tabularx}
{Estimativa de custo por execução em GPU única (Vast.ai, P25), a partir dos tempos reportados nos logs. Os valores são aproximações para comparabilidade: o custo efetivamente pago pode variar conforme o preço/h da instância no momento da execução e tempos ociosos fora do job.}

\end{table}

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}X|c|c|}
\hline
**Componente** & **Tempo (h)** & **Custo (USD)** \\
\hline
CPT-QLoRA (BrWaC 10k) & 2.35 & 0.71 \\
SFT-QLoRA (Canarim 10k) & 1.06 & 0.32 \\
Avaliação extrínseca (3 seeds) & 1.81 & 0.54 \\
\hline
**Total Trilha 2 + avaliação** & **5.22** & **1.57** \\
\hline
\end{tabularx}
{Custo total da Trilha 2 (treinos) e da avaliação extrínseca, em 1x RTX 4090 (Vast.ai, P25).}

\end{table}

### Estratégia Ótima: Fine-Tuning de Modelos Pré-Treinados

#### Análise Comparativa de Abordagens

Dadas as limitações identificadas, a estratégia de fine-tuning (pós-treinamento eficiente via LoRA/QLoRA) emerge como a abordagem mais viável em GPU única. A Tabela~ resume as principais alternativas e sua viabilidade em 24~GB de VRAM.

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{4.2cm}|c|c|>{\raggedright\arraybackslash}X|}
\hline
**Abordagem** & **VRAM (GB)** & **Viável?** & **Observação** \\
\hline
Treinamento do zero (7B) & $\approx 112$ & Não & requer paralelismo/otimizações (ZeRO, multi-GPU) \\
Continued pretraining (CPT) 8B + LoRA/QLoRA & $\le 24$ & Sim & texto bruto em PT-BR (BrWaC) \\
Supervised fine-tuning (SFT) 8B + LoRA/QLoRA & $\le 24$ & Sim & pares instrução--resposta (PT-BR) \\
QLoRA em modelos muito maiores ($\gg$ 8B) & $> 24$ & Depende & pode exigir múltiplas GPUs e/ou offloading \\
\hline
\end{tabularx}
{Viabilidade resumida de estratégias de desenvolvimento em GPU única (RTX 4090, 24 GB).}

\end{table}

#### CPT e SFT: definição e papel na adaptação

Nesta monografia, a trilha de pós-treinamento é decomposta em duas etapas conceitualmente distintas:

    \item \textbf{*Continued Pretraining* (CPT):} continuação do pré-treinamento com o mesmo objetivo auto-supervisionado do modelo base (predição do próximo token), porém com dados direcionados ao idioma/domínio de interesse (e.g., amostras do BrWaC em PT-BR). O CPT busca adaptar a distribuição linguística do modelo sem impor um formato de instruções [gururangan2020dont].
    \item \textbf{*Supervised Fine-Tuning* (SFT):} ajuste supervisionado em pares instrução--resposta (ou formatos *chat*), visando comportamento útil e aderente a comandos do usuário. No SFT, o modelo aprende padrões de seguimento de instrução e estrutura de resposta a partir de exemplos rotulados [ouyang2022training].


Do ponto de vista computacional, CPT e SFT podem ser executados com as mesmas técnicas de adaptação eficiente: **LoRA** injeta matrizes de baixo-rank em camadas alvo para treinar um pequeno subconjunto de parâmetros [hu2021lora], enquanto **QLoRA** combina LoRA com quantização (NF4) dos pesos do modelo base para reduzir VRAM e viabilizar sequências mais longas em GPU única [dettmers2023qlora]. Assim, a diferença central entre CPT e SFT está no *tipo de dado e objetivo experimental* (texto bruto vs. instruções), e não no mecanismo de otimização em si.

#### Fine-Tuning com LoRA: Análise Matemática

Low-Rank Adaptation (LoRA) reduz drasticamente os requisitos de memória ao parametrizar atualizações como matrizes de baixo rank:

\begin{equation}
W' = W + \Delta W = W + BA
\end{equation}

onde $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, e $r \ll \min(d,k)$.
\noindent Intuitivamente, a atualização $\Delta W = BA$ é dita de *baixo rank* porque a dimensão interna $r$ é pequena, o que permite representar a adaptação como o produto de duas matrizes ``finas'' (reduzindo parâmetros treináveis e consumo de memória).

**Redução de parâmetros treináveis:**
Para modelo Llama 3 8B com $r = 64$:
\begin{align}
\text{Parâmetros}_{originais} &= 8 \times 10^9 \\
\text{Parâmetros}_{LoRA} &= \sum_{i} r_i(d_i + k_i) \approx 16 \times 10^6 \\
\text{Redução} &= \frac{8 \times 10^9}{16 \times 10^6} = 500x
\end{align}

**Requisitos de memória para LoRA:**
\begin{align}
\text{Memória}_{\text{base}} &= 8 \times 10^9 \times 2 \text{ bytes (FP16)} = 16 \text{ GB} \\
\text{Memória}_{\text{LoRA}} &= 16 \times 10^6 \times 4 \text{ bytes (FP32)} = 64 \text{ MB} \\
\text{Memória}_{\text{gradientes}} &= 64 \text{ MB} \\
\text{Memória}_{\text{otimizador}} &= 128 \text{ MB} \\
\text{Memória}_{\text{total}} &\approx 16.3 \text{ GB por GPU}
\end{align}

### QLoRA para Modelos de Grande Escala

#### Quantização 4-bit com NormalFloat

QLoRA permite fine-tuning de modelos com dezenas de bilhões de parâmetros ao quantizar o modelo base para 4-bit (NF4) e treinar apenas adaptadores LoRA. O valor ``70B'' é usado aqui apenas como **exemplo de ordem de grandeza** (um tamanho comum em famílias abertas), e os cálculos escalam aproximadamente de forma linear com o número de parâmetros:

**Redução de memória base:**
\begin{align}
\text{Memória}_{\text{FP16}} &= 70 \times 10^9 \times 2 = 140 \text{ GB} \\
\text{Memória}_{\text{NF4}} &= 70 \times 10^9 \times 0.5 = 35 \text{ GB} \\
\text{Redução} &= \frac{140}{35} = 4x
\end{align}

**Requisitos totais (exemplo 70B):**
\begin{align}
\text{Memória}_{\text{base quantizada}} &= 35 \text{ GB} \\
\text{Memória}_{\text{adapters LoRA}} &= 200 \text{ MB} \\
\text{Memória}_{\text{gradientes}} &= 200 \text{ MB} \\
\text{Memória}_{\text{overhead}} &= 8 \text{ GB} \\
\text{Memória}_{\text{total}} &\approx 43.4 \text{ GB}
\end{align}

Em uma GPU única de 24 GB, QLoRA amplia significativamente a viabilidade de pós-treinamento. Ainda assim, modelos muito grandes (na faixa de dezenas de bilhões) podem exigir múltiplas GPUs e/ou *offloading*; no contexto deste trabalho, o alvo prático foi o regime 8B em 24 GB (Trilha 2), onde CPT/SFT tornam-se executáveis com baixo custo.

## Metodologia de Implementação

### Pipeline de Pré-processamento de Dados

#### Fonte de dados e exportação do corpus (BrWaC)

A base textual utilizada na trilha experimental (v1--v6) foi o corpus BrWaC, carregado via HuggingFace Datasets. Para treinamento do tokenizer, uma amostra embaralhada foi exportada para um arquivo de texto (`corpus\_v6.txt`), após limpeza com regras de qualidade (por exemplo: remoção de linhas muito curtas e linhas com baixa proporção de caracteres alfabéticos), preservando caixa e números para manter características do português brasileiro.

#### Treinamento do tokenizer (SentencePiece)

SentencePiece é uma biblioteca amplamente usada para treinar tokenizers *subword* diretamente sobre texto cru, com algoritmos como *unigram* e BPE. Neste trabalho, o tokenizer foi treinado com SentencePiece no modo *unigram*, com vocabulário de 16k subpalavras, gerando o modelo (`.model`) e estatísticas auxiliares para análise de fragmentação. Essa etapa é crítica para reduzir *byte fallback* e evitar excesso de tokens curtos em palavras comuns do português.

#### Datasets para pós-treinamento (CPT e SFT)

Para a trilha de pós-treinamento eficiente, os datasets foram definidos para manter comparabilidade de custo:

    \item **CPT:** amostra de 10k documentos do BrWaC (texto bruto), com `seq\_len`=2048.
    \item **SFT:** 10k exemplos do Canarim-Instruct PT-BR (dataset instrucional em português brasileiro, com pares instrução--resposta), com split 80/10/10 (treino/val/teste) e `seq\_len`=2048.


\begin{algoritmo}
{Pipeline de preparação de dados e treinamento (visão geral)}


    \item Carregar e amostrar dados (BrWaC / Canarim) com semente fixa.
    \item Limpar/normalizar textos e exportar corpus para tokenizer (quando aplicável).
    \item Treinar tokenizer SentencePiece e inspecionar métricas de fragmentação.
    \item Treinar o modelo (treino do zero ou LoRA/QLoRA), registrando logs e checkpoints.
    \item Gerar amostras e aplicar *guardrails* para validação qualitativa.

\end{algoritmo}

### Configuração do Ambiente de Fine-Tuning

#### Restrição prática: GPU única de consumo

O objetivo deste projeto é propor um caminho viável mesmo fora de ambientes corporativos, portanto a configuração de pós-treinamento foi desenhada para caber em uma única GPU de consumo. Neste trabalho, o ambiente de referência para os experimentos de pós-treinamento (CPT/SFT) e para o release candidate v6 é uma NVIDIA RTX 4090 (24 GB), e a técnica preferencial é LoRA/QLoRA para reduzir custo de memória e tempo.

#### Configuração alvo (Llama 3.1--8B + LoRA/QLoRA)


    \item **Modelo base**: Llama 3.1--8B (*base* para CPT; *instruct* para SFT).
    \item **Técnica**: LoRA/QLoRA (NF4) como padrão para permitir janelas maiores de contexto com 24 GB de VRAM.
    \item **Comprimento de sequência**: `seq\_len`=2048.
    \item **Datasets**: BrWaC (CPT) e Canarim-Instruct PT-BR (SFT), ambos com subconjuntos de 10k amostras para manter comparabilidade de custo.
    \item **Split**: 80/10/10 (treino/val/teste) e registro de tempo total, throughput e versões de software/hardware.


#### Otimizações de memória e estabilidade


    \item **Gradient accumulation** para ajustar *batch* efetivo sem exceder VRAM.
    \item **Gradient checkpointing** para reduzir memória de ativações em sequências longas.
    \item **Checkpoints e logs** para recuperação após falhas e auditoria dos resultados.


### Cronograma e Estimativas de Tempo

#### Análise Temporal Detalhada

Em vez de depender de estimativas abstratas, este trabalho prioriza tempos observados e registrados em logs. A trilha do zero (v1--v6) produz medições diretas de custo em GPU para um protótipo treinado localmente, enquanto a trilha LoRA/QLoRA é reportada com o mesmo rigor (tempo total, throughput e configuração de hardware/software), permitindo comparações justas sob a mesma restrição de orçamento computacional.

#### Análise de Riscos e Mitigações

**Riscos técnicos identificados:**

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.2cm}|>{\arraybackslash}p{2.4cm}|>{\arraybackslash}p{1.8cm}|>{\raggedright\arraybackslash}X|}
\hline
**Risco** & **Probabilidade** & **Impacto** & **Mitigação** \\
\hline
Falha de GPU & Média & Alto & Reinstanciar em outro host e retomar a partir de checkpoints \\
Overflow de memória & Baixa & Alto & Gradient checkpointing \\
Instabilidade de rede & Baixa & Médio & Checkpoints frequentes \\
Qualidade de dados & Média & Médio & Validação rigorosa \\
\hline
\end{tabularx}
{Matriz de riscos do projeto}
\end{table}

## Avaliação e resultados

### Linha experimental (v1--v6): evolução, limitações e resultados

Esta seção consolida os resultados das versões v1--v6 do protótipo, destacando custo computacional, limitações e descobertas ao longo das iterações. Os logs e artefatos experimentais (modelos, mapeamentos, históricos de treino, amostras e relatórios) estão disponíveis no repositório do projeto, com destaque para os diretórios `analysis/` e `versions/`.

#### Protocolo de avaliação e comparabilidade

Ao longo do processo, a tokenização e a arquitetura mudaram (caractere $\rightarrow$ subword $\rightarrow$ Transformer), portanto as métricas numéricas entre versões não são diretamente comparáveis de forma estrita. Por esse motivo, os resultados são reportados em três camadas:

    \item **Avaliação padronizada por janelas (v1--v4):** métricas em um conjunto fixo de janelas para comparação dentro de cada fase (char-level e subword), registradas em [analysis/artifacts/results/results_char_models_20k.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/analysis/artifacts/results/results_char_models_20k.json) e [analysis/artifacts/results](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/analysis/artifacts/results) (`results_subword_models_*.json`).
    \item **Histórico de treino (v5--v6):** melhores métricas intrínsecas de validação registradas nos logs ([versions/v5-transformer/logs/train_v5_main.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v5-transformer/logs/train_v5_main.json) e [versions/v6-release-candidate/logs](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/v6-release-candidate/logs) (`train_*.json`)).
    \item **Evidência qualitativa e guardrails:** amostras geradas e heurísticas de fragmentação para capturar falhas que perda/perplexidade não penalizam adequadamente.


\medskip
\noindent**Loss e perplexidade (PPL):** Neste trabalho, *loss* refere-se à perda de validação (cross-entropy média por token; menor é melhor). A perplexidade é reportada como $\mathrm{PPL}=e^{loss}$; valores menores indicam maior probabilidade atribuída ao próximo token. Como a unidade ``token'' depende do tokenizer, loss/PPL só são diretamente comparáveis quando a tokenização é a mesma; por isso, comparações entre tokenizers distintos (e.g., v5 *vs.* v5b *vs.* v6) devem ser interpretadas com cautela.
\medskip

#### Protocolo de avaliação (Trilha 2: CPT/SFT-QLoRA)

Enquanto a linha experimental v1--v6 prioriza avaliação intrínseca e inspeção qualitativa, a Trilha 2 (CPT/SFT via LoRA/QLoRA) requer avaliação extrínseca em tarefas, comparando modelos base vs.\ pós-treinados sob o mesmo orçamento computacional. O protocolo executado é detalhado a seguir, e seus resultados consolidados estão na Tabela~.


\medskip
\noindent**Objetivos e hipóteses;**

    \item **Objetivo:** demonstrar ganho consistente do SFT em PT-BR em tarefas de QA, sumarização e reescrita, versus baseline instruccional zero-shot.
    \item **Hipóteses:** (i) dados PT-BR + LoRA [hu2021lora] melhoram coerência/estilo local; (ii) CPT em PT-BR reduz fricção linguística e melhora consistência lexical; (iii) avaliação por múltiplas seeds e rubricas reduz viés de análise qualitativa.


\medskip
\noindent**Tarefas e conjuntos de teste;**

    \item **QA factual curto (PT-BR):** amostra filtrada do Canarim com referências curtas (`n=200`).
    \item **Sumarização (PT-BR):** amostra filtrada do Canarim (`n=200`).
    \item **Reescrita (PT-BR):** amostra filtrada do Canarim (`n=200`).
    \item **Construção dos conjuntos:** os subconjuntos foram gerados em [versions/trilha2-lora/analysis/extrinsic](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/trilha2-lora/analysis/extrinsic), com exclusão de overlap com o subconjunto usado no SFT:
    
        \item [versions/trilha2-lora/datasets/canarim_sft_10k_train.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/datasets/canarim_sft_10k_train.jsonl)
        \item [versions/trilha2-lora/datasets/canarim_sft_10k_val.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/datasets/canarim_sft_10k_val.jsonl)
        \item [versions/trilha2-lora/datasets/canarim_sft_10k_test.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/datasets/canarim_sft_10k_test.jsonl)
    
    via [versions/trilha2-lora/scripts/09_prepare_extrinsic_eval_sets.py](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/scripts/09_prepare_extrinsic_eval_sets.py).


\medskip
\noindent**Baselines e comparações;**

    \item **Baseline-A:** LLaMA~3.1-8B-Instruct (zero-shot).
    \item **Baseline-B (opcional):** LLaMA~3.1-8B + CPT-QLoRA (BrWaC 10k), reportado por métricas intrínsecas; não incluído na avaliação extrínseca por não ser um modelo instruccional.
    \item **Modelo alvo:** LLaMA~3.1-8B-Instruct + SFT-QLoRA (Canarim 10k).


\medskip
\noindent**Métricas e protocolo;**

    \item **QA:** EM (*exact match*) e F1 (sobreposição lexical entre predição e referência), em respostas curtas; normalização de caixa/acentos.
    \item **Sumarização e reescrita:** ROUGE-L (F1), baseado na maior subsequência comum (*LCS*), como métrica automática comparável.
    \item **Inferência:** geração determinística (`do\_sample=false`, `temperature=0`, `top\_p=1`), com registro de throughput/custo via logs estruturados.
    \item **Robustez:** repetição em 3 seeds de amostragem do conjunto de teste (123/456/789), reportando média$\pm$desvio padrão.
    \item **Robustez/segurança (opcional):** checagens simples (alucinação, PII, prompt injection) e registro de limitações.


\medskip
\noindent**Tabela de resultados;**
Os resultados consolidados da avaliação extrínseca estão apresentados na Tabela~. Os resumos e o agregado multi-seed estão em [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json); os detalhes por exemplo (saídas e métricas) estão versionados em [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl), [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed456.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed456.jsonl) e [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed789.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed789.jsonl), gerados por [versions/trilha2-lora/scripts/10_run_extrinsic_eval.py](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/scripts/10_run_extrinsic_eval.py).

\medskip
\noindent**Infra, custos e reprodutibilidade;**

    \item **Setup:** 1$\times$~NVIDIA RTX 4090 (24 GB) via Vast.ai; LoRA/QLoRA [dettmers2023qlora]; checkpoints frequentes.
    \item **Métricas de eficiência:** tokens/s, GPU-h e custo estimado (USD), além de limites de VRAM/IO.
    \item **Reprodutibilidade:** seeds, versões (CUDA/driver/transformers/peft/bitsandbytes), scripts/configs versionados e logs estruturados com metadados.


\medskip
\noindent**Execução e artefatos;**
A Trilha 2 foi executada no nível de treinos (CPT e SFT) e de avaliação extrínseca (QA, sumarização e reescrita) no Canarim filtrado, com resultados e artefatos versionados em [versions/trilha2-lora](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/trilha2-lora).


#### Resultados v1--v3 (char-level)

\begin{table}[ht]

\begin{tabular}{|l|l|c|c|c|c|c|}
\hline
**Versão** & **Arquitetura** & **Tempo (h)** & **Tokens est. (B)** & **Loss** & **PPL** & **Acc.** \\
\hline
v1 & CharRNN (LSTM) & 1.1 & 13.88 & 1.522 & 4.58 & 55.0\% \\
v2 & CharLM (LSTM) & 0.9 & 3.47 & 1.437 & 4.21 & 56.7\% \\
v3 & Stacked LSTM & 2.0 & 3.47 & 1.375 & 3.96 & 58.2\% \\
\hline
\end{tabular}
{Resultados de avaliação padronizada na fase char-level (v1--v3).}

\end{table}

Nesta fase, o objetivo foi construir o pipeline e entender limites de custo/engenharia. Apesar dos ganhos graduais em métricas, observou-se baixa qualidade textual em português (acentos e segmentação), o que motivou a mudança para subword na v4.

#### Resultados v4 (subword + RNN)

\begin{table}[ht]

\begin{tabular}{|l|l|c|c|c|c|c|}
\hline
**Execução** & **RNN** & **Vocab** & **Épocas** & **Tempo (h)** & **Tokens est. (B)** & **PPL** \\
\hline
`v4\_subword` & GRU & 3.2k & 2 & 1.2 & 2.39 & 75.83 \\
`v4\_subword\_ep6` & GRU & 3.2k & 6 & 3.7 & 7.18 & 72.73 \\
`v4k\_bf\_ep2` & GRU & 4k & 2 & 1.2 & 2.32 & 84.31 \\
`v4k\_bf\_lstm\_ep2` & LSTM & 4k & 2 & 1.4 & 2.32 & 85.37 \\
\hline
\end{tabular}
{Resultados de avaliação padronizada na fase subword + RNN (v4).}

\end{table}

A introdução de subwords (SentencePiece) foi o principal salto qualitativo desta etapa: frases passaram a ser mais legíveis e a tokenização reduziu a necessidade de ``soletrar'' caracteres. Ao mesmo tempo, os experimentos com vocabulário e fallback evidenciaram um novo desafio: escolhas de tokenizer afetam fortemente a fragmentação e podem degradar a geração mesmo quando as métricas intrínsecas não parecem alarmantes.

#### Resultados v5--v6 (Transformer)

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.2cm}|>{\raggedright\arraybackslash}p{2.4cm}|c|c|c|c|}
\hline
**Execução** & **Tokenizer** & **Tempo (h)** & **Tokens (B)** & \textbf{Melhor `val\_loss`} & \textbf{Melhor `val\_ppl`} \\
\hline
`v5_main` & 4k (+fallback) & 3.2 & 6.00 & 3.8918 & 49.00 \\
`v5b_long` & unigram 12k & 2.9 & 0.66 & 3.3843 & 29.50 \\
`v6_rc16k_v5b` & unigram 16k & 13.3 & 3.76 & 3.3001 & 27.12 \\
`v6_rc16k_v5b_aligned` & unigram 16k & 16.6 & 3.63 & 3.2935 & 26.94 \\
\hline
\end{tabularx}
{Resumo das execuções com Transformer (v5--v6). Métricas de perplexidade não são diretamente comparáveis entre tokenizers.}

\end{table}

Na v5, observou-se um fenômeno central para este trabalho: mesmo com melhora clara em `val\_loss`/`val\_ppl`, a qualidade percebida do texto degradou (fragmentação em subwords e artefatos visuais). Isso motivou duas mudanças metodológicas: (i) revisão do tokenizer (v5b) e (ii) inclusão de *guardrails* automatizados na avaliação de amostras.

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.2cm}|>{\arraybackslash}p{2.6cm}|>{\arraybackslash}p{3.2cm}|c|}
\hline
**Execução** & \textbf{\shortstack{`fallback\_`\\`ratio`}} & \textbf{\shortstack{`short\_piece\_`\\`ratio`}} & **Passa?** \\
\hline
`v5b_long_final` & 0.000 & 0.050 & Sim \\
`v6_rc16k_v5b` & 0.000 & 0.149 & Não \\
`v6_rc16k_v5b_aligned` & 0.000 & 0.155 & Não \\
\hline
\end{tabularx}
{Guardrails de tokenização em amostras geradas (5 sementes por execução).}

\end{table}

Observa-se uma melhora pequena em `val\_loss`/`val\_ppl` após o alinhamento da limpeza entre corpus do tokenizer e treino; entretanto, o `short\_piece\_ratio` permaneceu acima do limiar definido, sugerindo que a fragmentação está mais relacionada à configuração do vocabulário/tokenizer do que exclusivamente à limpeza.

#### Exemplos qualitativos (amostras geradas)

Para complementar as métricas, apresenta-se a seguir um recorte de amostras geradas com **prompts padronizados** (os mesmos em todas as versões). Os trechos foram truncados apenas para facilitar leitura no documento; as amostras completas estão disponíveis em:

    \item [analysis/artifacts/samples/samples_brwac_custom_prompts_utf8.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/analysis/artifacts/samples/samples_brwac_custom_prompts_utf8.json) (v1--v4)
    \item [analysis/artifacts/samples/samples_v5_main.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/analysis/artifacts/samples/samples_v5_main.json)
    \item [analysis/artifacts/samples/samples_v5b_long_final_tp075.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/analysis/artifacts/samples/samples_v5b_long_final_tp075.json)
    \item [versions/v6-release-candidate/analysis/samples_v6_rc16k_v5b_aligned.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/v6-release-candidate/analysis/samples_v6_rc16k_v5b_aligned.json)


\bigskip
\noindent**Prompt 1 (Infraestrutura).**

\begin{tccprompt}
PROMPT:
O setor de infraestrutura logistica brasileira debate novas concessoes ferroviarias,
cronogramas de duplicacao, metas de produtividade e integracao com portos para desafogar
corredores de exportacao.

v1:
a matéria e as empresas e do conteúdo de gente como o mundo a contratos e reverter para
concentração de encontro de papel e de acordo no estado de ator com a rede de alguns
melhores palavras do projeto do seguro de um conteúdo do direito com o povo com a maior de
um pacacionaliz

v2:
como ter uma tentativa de mortes de tradicionalização de uma base de modo que se interessa a
escolha de legislatura e constituinte de uma marca de crime infecto de transporte com a
conceitualidade. em 0 o regime de vista é que o destaque com o processo de estado com pai a
princi

v3:
a matéria especializada pode ser desprezada com o texto ativo do projeto de verdadeiro
conjunto de processos promovidos na consciência externa de responsabilidade do computador do
país a considerar o então de uma apresentação de design assistir com a escola informada. a
abordage

v4 (v4k_bf_ep2):
- o seu nome se refere à borda de uma vida em círculo, é o que se o mais alto, para o
momento, o que quer, com o seu papel, o que o sol não se é o maior que se pode. o
sequestramento de cavaleiros e a cápula, o que é, no sentido de passar,, de modo que o seu
seu papel não se transforma em seu corpo e é o seu maior valo [...]

v5 (main):
clín informações viagem claro ta vê encamatos reserv suasóveisõe obrigu concre usar ecoên
clubeson marca pen? áre visão servidoresicidade anal 1. peso num prefeitoílias comput
sobfeito prazo banda autom total deseist flitasir suturatados poss tema manterbas plapenho?
serem intro? debate ag minutos_?terelhibil jovenssp [...]

v5b (final):
O setor de infraestrutura logistica brasileira debate novas concessoes ferroviarias,
cronogramas de duplicacao, metas de produtividade e integracao com portos para desafogar
corredores de exportacao. projeto de lei no 0/0 - de 0/0/0 - dou 0/0/0 altera o artigo 0 da
lei no 0/0, que dispõe sobre a concessão de parcelamen [...]

v6 (aligned):
Segundo o secretário de Segurança Pública, Francisco Teixeira, a tendência é que o setor de
infraestrutura dê continuidade à implementação. "Na realidade, os pequenos negócios se
intensem em mais de um terço do mercado no futuro. Já no cenário do setor, as esferas
produtivas podem ser divididas entre as economias exter [...]

\end{tccprompt}

\medskip
\noindent**Prompt 2 (Futebol).**

\begin{tccprompt}
PROMPT:
A diretoria de um clube de futebol negocia patrocinio master, reforcos na janela de
transferencia, metas de bilheteria, programa de socios e auditoria das contas apos uma
temporada irregular nos gramados.

v1:
na estrutura de conta as contratados com a importância da vida para trabalho de motor como a
democratica e comparação com a criança de se presente são a diretoria do amor na agosto, no
manutenção da extra a via tomar que foram de energia ao serviço do processo em um estado de
co

v2:
o cerca de 0 anos e composição da fila de trabalho no presidente do trabalho de produção e
condições de responsável pelo processo de produção de seu primeiro caso de 0 de maio de 0, o
ministério de resultados adversários e três anos de previsão e proporcionalidade, a primeira
di

v3:
o contrato que está como a proposta de despesas de projetos de serviços de instituição de
energia de energia do trabalho e da educação no presidente da cidade de polícia federal e
profissionais e paraná a expectativa do contrato de disposição de futebol, na administração
do curs

v4 (v4k_bf_ep2):
em 0, de 0 de março de 0, o primeiro cnb. o que se refere ao mesmo tempo, a pena de que o
brasil tenha sido feito no brasil. o ano passado é a melhor opção de hoje, com a
participação de um bom nível de atuação em quadrinhos. essa é o que você sabe se não gostar.
é muito importante para o mundo, um grande exemplo do pa [...]

v5 (main):
esfor interess passedtro imposntasconhec funçãoícios pposto fórédioch filme trabalh minist
negócios julho gal sen próxima? altura fundam op dese pec fevereiro micro rura melhoícia
próximaencia mostra línur?agemiadotou ouv determinvalío poderia produz sãobra dr dela
tamanho econôm eduardo figuro facm leirações interes i [...]

v5b (final):
A diretoria de um clube de futebol negocia patrocinio master, reforcos na janela de
transferencia, metas de bilheteria, programa de socios e auditoria das contas apos uma
temporada irregular nos gramados. o presidente do clube, roberto marinho, disse que a saída
de bola foi "um pouco mais dura" e que o time pode ter um [...]

v6 (aligned):
Para os campeonatos em Fortaleza, a diretoria só tem uma semana e meio para chegar no fim,
enquanto o mesmo está sendo negociado no final de semana. O Atlético-MG enfrenta o Vitória
no último domingo (30), diante do Fluminense por 2 a 1. Com 16 pontos e sete pontos, o
Vitória recebe o Vitória-BA neste sábado (31) e o F [...]

\end{tccprompt}


#### Limitações, desafios e descobertas


    \item **Métricas não garantem legibilidade:** a v5 mostrou que reduzir `val\_loss`/`val\_ppl` não implica, necessariamente, melhora em fluência e integridade de palavras; por isso, a avaliação passou a incluir guardrails e inspeção sistemática de amostras.
    \item **Tokenização é componente crítico:** a transição char-level $\rightarrow$ subword (v4) e a ampliação de vocabulário (v5b) tiveram impacto qualitativo maior do que ajustes marginais de arquitetura/hiperparâmetros em fases anteriores.
    \item **Limitações computacionais moldam o método:** a necessidade de operar em GPU única impôs modelos menores e janelas limitadas, reforçando a estratégia de (i) prototipagem do zero em escala reduzida e (ii) pós-treinamento eficiente (CPT/SFT) como alternativa prática.
    \item **Alinhamento de limpeza tem efeito limitado:** no v6, alinhar a limpeza do corpus do tokenizer com a do treino trouxe ganho marginal nas métricas, mas não resolveu fragmentação medida pelos guardrails.


### Trilha 2 (QLoRA): CPT e SFT em PT-BR

Nesta trilha, investiga-se pós-treinamento eficiente (LoRA/QLoRA) em um modelo base open-source de maior capacidade, com dois estágios: (i) *continued pretraining* (CPT) em português (BrWaC) e (ii) *supervised fine-tuning* (SFT) com dados instrucionais (Canarim-Instruct PT-BR). O objetivo é demonstrar ganhos mensuráveis com baixo custo, mantendo rastreabilidade por meio de logs estruturados e artefatos versionados em [versions/trilha2-lora](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/tree/main/versions/trilha2-lora).

#### Resumo dos treinamentos

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}X|>{\raggedright\arraybackslash}p{3.0cm}|>{\raggedright\arraybackslash}p{1.8cm}|c|c|>{\arraybackslash}p{2.6cm}|}
\hline
**Execução** & **Modelo base** & **Dados** & **Tempo (h)** & **Custo (USD)** & \shortstack{\textbf{Melhor `eval\_loss`}\\\textbf{(PPL $e^{loss}$)}} \\
\hline
[versions/trilha2-lora/logs/train_cpt_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_cpt_qlora.json) & LLaMA~3.1-8B & BrWaC 10k & 2.35 & 0.71 & 2.140 (8.50) \\
[versions/trilha2-lora/logs/train_sft_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_sft_qlora.json) & LLaMA~3.1-8B-Instruct & Canarim 10k & 1.06 & 0.32 & 1.104 (3.02) \\
\hline
\end{tabularx}
{Resumo dos treinamentos da Trilha 2 em 1x RTX 4090 (24 GB), com `seq\_len`=2048.}

\end{table}

Os logs completos estão em [versions/trilha2-lora/logs/train_cpt_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_cpt_qlora.json) e [versions/trilha2-lora/logs/train_sft_qlora.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/logs/train_sft_qlora.json).

#### Avaliação intrínseca comparável (base vs.\ pós-treinado)

Para reduzir o risco de interpretações equivocadas em qualitativos fora do domínio, reporta-se a seguir uma avaliação intrínseca comparável: (i) no CPT, perda/perplexidade em split de validação do BrWaC amostrado; (ii) no SFT, perda calculada apenas sobre os tokens de resposta (os tokens do prompt são mascarados no cálculo da loss) no split de validação do Canarim. A execução consolidada está em [versions/trilha2-lora/analysis/eval_trilha2_baselines.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_baselines.json).

\begin{table}[ht]

\scriptsize
\begin{tabular}{|l|l|c|c|c|}
\hline
**Comparação** & **Split de avaliação** & \textbf{`eval\_loss`} & \textbf{PPL ($e^{loss}$)} & **Ganho** \\
\hline
LLaMA~3.1-8B (base) & BrWaC (val.) & 2.372 & 10.72 & -- \\
+ CPT-QLoRA & BrWaC (val.) & 2.159 & 8.66 & $-19.2\%$ (PPL) \\
\hline
LLaMA~3.1-8B-Instruct (base) & Canarim (val.) & 1.778 & 5.92 & -- \\
+ SFT-QLoRA & Canarim (val.) & 1.117 & 3.06 & $-48.4\%$ (PPL) \\
\hline
\end{tabular}
{Avaliação intrínseca (loss e perplexidade equivalente) para base vs.\ adaptado em CPT e SFT.}

\end{table}

#### Exemplos qualitativos (SFT-QLoRA em Canarim)

Para complementar as métricas intrínsecas, apresenta-se abaixo um recorte de exemplos do **split de validação** do Canarim, escolhidos para ilustrar casos em que o SFT melhora a aderência ao *formato esperado* (respostas curtas e diretamente avaliáveis).
Durante a inspeção, observou-se ruído em parte das referências do dataset, o que reforça a importância de filtros de qualidade e curadoria na etapa de SFT.

**Exemplo 1 (classificação):**

\begin{tccprompt}
INSTRUÇÃO:
Classifique o sexo do autor. Artigo: A melhor maneira de cozinhar um bife na grelha.

BASE (LLaMA 3.1-8B-Instruct):
Não é possível determinar o sexo do autor com base no artigo fornecido, pois não há
informações sobre a identidade do autor. O artigo parece ser um texto informativo sobre
uma receita de cozinha, mas não fornece informações sobre a identidade do autor.

SFT-QLoRA (Canarim 10k):
Homem
\end{tccprompt}

**Exemplo 2 (classificação de impacto):**

\begin{tccprompt}
INSTRUÇÃO:
Tarefa: Diga-me se uma notícia é positiva ou negativa para um preço de ação da empresa.
Notícia: O CEO da empresa está deixando sua posição.

BASE (LLaMA 3.1-8B-Instruct):
Essa notícia é negativa para o preço da ação da empresa. A saída do CEO pode ser vista
como um sinal de instabilidade e incerteza sobre o futuro da empresa, o que pode afetar
negativamente a confiança dos investidores e, consequentemente, o preço da ação.

SFT-QLoRA (Canarim 10k):
Negativo
\end{tccprompt}

#### Avaliação extrínseca (QA, sumarização e reescrita)

Para evidência mais direta de utilidade prática, executou-se uma avaliação extrínseca em três tarefas (QA, sumarização e reescrita) usando um conjunto de teste filtrado a partir do Canarim (`n=200` por tarefa), com exclusão de overlap com o subconjunto usado no SFT (train/val/test). A geração foi determinística (`do\_sample=false`, `temperature=0`) e a avaliação foi repetida em **3 seeds de amostragem** do conjunto de teste (123/456/789), reportando média$\pm$desvio padrão. Os artefatos estão em [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_multiseed.json) (agregado), com detalhes por execução em [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl), [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed456.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed456.jsonl) e [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed789.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details_seed789.jsonl).

\begin{table}[ht]

\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}X|c|>{\raggedright\arraybackslash}p{2.8cm}|c|c|>{\raggedright\arraybackslash}p{2.8cm}|}
\hline
**Tarefa** & **n** & **Métrica** & **Baseline** & **SFT-QLoRA** & **Ganho** \\
\hline
QA (resposta curta) & 200 & EM & 0.037$\pm$0.013 & 0.090$\pm$0.017 & +0.053 (+145\%) \\
QA (resposta curta) & 200 & F1 & 0.063$\pm$0.018 & 0.113$\pm$0.018 & +0.050 (+80\%) \\
Sumarização & 200 & ROUGE-L (F1) & 0.154$\pm$0.004 & 0.204$\pm$0.010 & +0.050 (+33\%) \\
Reescrita & 200 & ROUGE-L (F1) & 0.173$\pm$0.021 & 0.295$\pm$0.020 & +0.123 (+71\%) \\
\hline
\end{tabularx}
{Avaliação extrínseca em Canarim filtrado (média$\pm$desvio padrão em 3 seeds de amostragem): baseline LLaMA~3.1-8B-Instruct vs.\ LLaMA~3.1-8B-Instruct + SFT-QLoRA (Canarim 10k), com geração determinística. Custo de inferência estimado (somado nas 3 execuções): 1.81 GPU-h (US\$ 0.54) em 1x RTX 4090.}

\end{table}

#### Exemplos qualitativos (avaliação extrínseca)

Para complementar a Tabela~, apresenta-se abaixo um recorte de três exemplos (um por tarefa) do conjunto de teste filtrado, ilustrando um padrão recorrente: após o SFT, as respostas tendem a ficar mais curtas e mais aderentes ao formato esperado pelo enunciado. Os trechos foram truncados apenas para facilitar leitura no documento; os detalhes por exemplo estão versionados em [versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl](https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora/blob/main/versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl).

**Exemplo 3 (QA):**

\begin{tccprompt}
INSTRUÇÃO:
Eu quero comprar uma guitarra elétrica, mas eu não tenho certeza qual escolher. Qual devo obter? (1: Squier Bullet Stratocaster; 2: Stratocaster padrão)

REFERÊNCIA:
1

BASE (LLaMA 3.1-8B-Instruct):
Ambas as opções são excelentes escolhas, mas dependem de suas preferências e necessidades específicas...

SFT-QLoRA (Canarim 10k):
1
\end{tccprompt}

**Exemplo 4 (Sumarização):**

\begin{tccprompt}
INSTRUÇÃO:
Encontre um artigo de notícias sobre o governo local e escreva um resumo dele. Artigo: O conselho da cidade de São Francisco votou para proibir a venda de peles dentro de seus limites. A proibição entrará em vigor em janeiro de 2020, e se aplicará a lojas de varejo e vendas on-line.

REFERÊNCIA (truncada):
O conselho da cidade de São Francisco votou a proibição da venda de peles dentro de seus limites... (Katy Tang)

BASE (LLaMA 3.1-8B-Instruct):
Infelizmente, não consegui encontrar um artigo de notícias específico... (segue com contexto incorreto e extrapolações)

SFT-QLoRA (Canarim 10k):
O conselho da cidade de São Francisco votou para proibir a venda de peles dentro de seus limites. A proibição entrará em vigor em janeiro de 2020, e se aplicará a lojas de varejo e vendas on-line.
\end{tccprompt}

**Exemplo 5 (Reescrita):**

\begin{tccprompt}
INSTRUÇÃO:
Descubra qual das seguintes frases é escrita em voz passiva e reescreva-a em voz ativa: “o gato foi comido pelo cachorro”.

REFERÊNCIA:
O cão comeu o gato.

BASE (LLaMA 3.1-8B-Instruct):
A frase está em voz passiva... Aqui está a reescrita em voz ativa: “O cachorro comeu o gato.” (com explicação adicional)

SFT-QLoRA (Canarim 10k):
O cão comeu o gato.
\end{tccprompt}


### Métricas de Performance

#### Benchmarks Específicos para Português

No escopo do protótipo v1--v6, a avaliação principal foi intrínseca (perda e perplexidade em validação) e por inspeção/guardrails em amostras geradas. As métricas abaixo representam benchmarks complementares desejáveis, especialmente relevantes para a etapa de pós-treinamento via LoRA/QLoRA.

**Avaliação quantitativa:**

    \item **BLEU score**: Tradução português-inglês
    \item **Perplexidade**: Corpus de teste brasileiro
    \item **ROUGE**: Sumarização de textos em português
    \item **Accuracy**: Classificação de sentimentos


**Benchmarks culturais específicos:**

    \item Conhecimento sobre geografia brasileira
    \item Compreensão de expressões regionais
    \item Conhecimento histórico e cultural
    \item Capacidade em domínios específicos (direito, medicina)


#### Comparação com Modelos Existentes

**Baseline models para comparação:**

    \item GPT-3.5 (OpenAI)
    \item Llama 3 8B base (Meta)
    \item Sabiá-7B (português brasileiro)
    \item BERTimbau (BERT português)


**Métricas de eficiência:**
\begin{align}
\text{Eficiência}_{computacional} &= \frac{\text{Performance}}{\text{FLOPs utilizados}} \\
\text{Eficiência}_{dados} &= \frac{\text{Performance}}{\text{Tokens de treinamento}} \\
\text{ROI}_{projeto} &= \frac{\text{Valor gerado}}{\text{Custo computacional}}
\end{align}

### Impacto e Aplicações

#### Aplicações Imediatas

Nesta monografia, a utilidade prática é evidenciada pela avaliação extrínseca da Trilha 2 (QA, sumarização e reescrita), que mede ganho em tarefas representativas de uso real em PT-BR sob baixo custo computacional.

**Potencial de impacto:**
Embora esteja fora do escopo deste trabalho quantificar benefícios econômicos com precisão, a motivação de soberania digital e utilidade prática se manifesta em aplicações que reduzem fricções linguísticas, aumentam produtividade e permitem maior controle sobre dados e modelos. A quantificação de impacto (e.g., ROI setorial) requer estudos dedicados e será tratada como trabalho futuro.

#### Contribuição para Soberania Digital

No contexto deste trabalho, a contribuição para soberania digital se materializa principalmente na viabilidade de adaptar modelos open-source ao PT-BR em hardware acessível (GPU única), aliada à rastreabilidade do pipeline (dados, scripts, logs e artefatos), o que facilita auditoria e reprodutibilidade.

#### Roadmap de expansão


    \item **Curto prazo** (6 meses): Modelo especializado funcional
    \item **Médio prazo** (1 ano): Família de modelos para diferentes domínios
    \item **Longo prazo** (2 anos): Infraestrutura nacional de IA


Esta discussão reforça que, apesar das limitações de hardware, é possível produzir evidência empírica do custo de uma linha ``do zero'' (em escala reduzida) e estruturar um caminho pragmático de pós-treinamento eficiente (LoRA/QLoRA) sobre modelos open-source, contribuindo para a discussão de soberania digital brasileira com base técnica e rastreabilidade.


## Síntese dos resultados

Esta monografia investigou, sob a perspectiva de soberania digital, o desafio prático de construir modelos de linguagem em português brasileiro sob restrições realistas de hardware. O trabalho foi estruturado em duas trilhas complementares: (i) uma linha experimental de treinamento do zero (v1--v6), em escala reduzida, para evidenciar custo e fragilidades do pipeline; e (ii) uma linha viável baseada em pós-treinamento eficiente (CPT/SFT com LoRA/QLoRA) sobre um modelo base open-source.

Na trilha v1--v6, o principal resultado é a evidência de que a dificuldade não é apenas ``ter uma GPU'', mas dominar um conjunto de decisões acopladas: limpeza do corpus, tokenização, definição de arquitetura, instrumentação do treinamento e critérios de avaliação. A evolução entre versões mostrou que:

    \item Alterações de tokenização podem produzir impactos qualitativos maiores do que melhorias marginais em perda/perplexidade, exigindo métricas complementares e inspeção sistemática de amostras.
    \item Métricas intrínsecas (e.g., `val\_loss`/`val\_ppl`) são úteis para monitorar convergência, mas não capturam, isoladamente, problemas de fragmentação e legibilidade em geração.
    \item Restrições de VRAM em GPU única tornam inviável treinar modelos de múltiplos bilhões de parâmetros do zero sem estratégias avançadas (paralelismo, offloading, otimizações de memória), reforçando o papel do pós-treinamento como estratégia pragmática.


Na Trilha 2, os experimentos de pós-treinamento eficiente via QLoRA apresentaram ganhos mensuráveis com baixo custo. Em particular, o SFT-QLoRA em Canarim 10k reduziu a perda intrínseca no split de validação (Tabela~) e, na avaliação extrínseca em tarefas (QA, sumarização e reescrita) com Canarim filtrado, apresentou melhorias consistentes versus o baseline instruccional (Tabela~). No total, CPT+SFT consumiram 3.41 horas de GPU (US\$ 1.03) e a avaliação extrínseca (3 seeds) 1.81 horas (US\$ 0.54), em 1x RTX 4090.

Do ponto de vista econômico, os experimentos também ilustram que protótipos em escala reduzida podem ser executados com baixo custo direto em infraestrutura sob demanda, mas isso não se traduz automaticamente em viabilidade de escalar para modelos de fronteira. A evidência de custo apresentada no Capítulo~ busca, sobretudo, tornar explícita a relação entre tempo de GPU, escolhas metodológicas e resultados obtidos.

## Limitações

As principais limitações do estudo são:

    \item **Comparabilidade entre tokenizers:** perdas e perplexidades não são diretamente comparáveis quando a tokenização muda, razão pela qual a análise combinou métricas intrínsecas com exemplos qualitativos e *guardrails* de fragmentação.
    \item **Avaliação centrada no protótipo:** na linha v1--v6, priorizou-se rastreabilidade e aprendizado de engenharia; métricas extrínsecas e avaliação humana permanecem como próximos passos para medir utilidade prática de forma mais direta.
    \item **Validade externa da avaliação extrínseca:** na Trilha 2, a avaliação em tarefas foi baseada em subconjuntos filtrados do Canarim, com métricas automáticas (EM/F1 e ROUGE-L). Apesar de ter sido evitado overlap com o subconjunto usado no SFT, ainda podem existir similaridades residuais e ruído nas referências, o que limita a generalização para benchmarks externos.


## Trabalhos futuros

Para consolidar a tese central de que o pós-treinamento eficiente oferece um caminho pragmaticamente viável no mesmo orçamento computacional, os próximos passos naturais são:

    \item **Benchmarks externos:** repetir a avaliação extrínseca em conjuntos padronizados (QA/sumarização/reescrita) para além do Canarim, com splits e rubricas fixas.
    \item **Robustez e variância:** avaliar múltiplas configurações de geração (seeds/temperatura) e reportar variância; complementar métricas automáticas com juiz/humano em amostras pequenas.
    \item **Ablations:** testar variações (por exemplo, incluir CPT-QLoRA como etapa intermediária) para isolar o efeito de CPT vs.\ SFT em tarefas downstream.
    \item **Robustez e segurança:** executar checagens simples (alucinação, PII, prompt injection) e registrar limitações.


Com isso, o documento passa a apresentar, de forma mais completa, tanto a evidência prática do custo de uma linha ``do zero'' quanto a evidência de uma alternativa mais eficiente para maximizar utilidade em português brasileiro sob restrições realistas de recursos.
