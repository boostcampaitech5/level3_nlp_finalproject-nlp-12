import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from collections import Counter
from transformers import AutoTokenizer


def barplot(data: pd.DataFrame, 
            column: str,
            column_order: list,
            title: str = "distribution",
            ylabel: str = 'count'):
    """
    pandas 데이터프레임의 특정 열에 대한 막대 그래프를 생성하고 표시하는 함수입니다.

    Args:
        data (pd.DataFrame): 그래프를 생성할 데이터프레임.
        column (str): 그래프를 생성할 데이터프레임의 열 이름.
        column_order (list): x축에 표시될 값들의 순서.
        title (str): 그래프의 제목. 기본값은 "distribution".
        ylabel (str): y축 레이블. 기본값은 "count".
    """

    plt.figure(figsize=(6,7))

    order = column_order if column_order else data[column].unique()
    ax = sns.countplot(x=column, data=data, order=order, palette='viridis')

    for p in ax.patches:
        ax.annotate(format(int(p.get_height())), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', 
                    va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')

    plt.title(title)
    plt.xlabel(str(column))
    plt.ylabel(ylabel)
    plt.show()


def dual_barplot(data: pd.DataFrame, 
            main_column: str,
            sub_column: str,
            column_order: list,
            title: str = "distribution",
            ylabel: str = 'count'):
    """
    pandas 데이터프레임의 두 열에 대해 이중 막대 그래프를 생성하고 표시하는 함수입니다.

    Args:
        data (pd.DataFrame): 그래프를 생성할 데이터프레임.
        main_column (str): 그래프를 생성할 주요 데이터프레임의 열 이름.
        sub_column (str): 그래프 내에서 색상으로 구분할 부분적인 데이터프레임의 열 이름.
        column_order (list): x축에 표시될 값들의 순서.
        title (str, 선택적): 그래프의 제목. 기본값은 "distribution".
        ylabel (str, 선택적): y축 레이블. 기본값은 "count".
    """

    plt.figure(figsize=(6,4))

    order = column_order if column_order else data[main_column].unique()
    ax = sns.countplot(x=main_column, data=data, hue=sub_column, order=order, palette='Blues')

    for p in ax.patches:
        ax.annotate(format(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', 
                    va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')

    plt.title(title)
    plt.xlabel(str(main_column))
    plt.ylabel(ylabel)
    plt.legend(title=str(sub_column))
    plt.show()


def pivot_barplot(data: pd.DataFrame, 
            main_column: str,
            sub_column: str,
            column_order: list,
            title: str = "distribution",
            ylabel: str = 'count'):
    """
    pandas 데이터프레임의 두 열을 이용하여 피봇 테이블을 생성하고, 이를 기반으로 한 스택형 막대 그래프를 생성하고 표시하는 함수입니다.

    Args:
        data (pd.DataFrame): 그래프를 생성할 데이터프레임.
        main_column (str): 그래프를 생성할 주요 데이터프레임의 열 이름.
        sub_column (str): 그래프 내에서 색상으로 구분할 부분적인 데이터프레임의 열 이름.
        column_order (list): x축에 표시될 값들의 순서.
        title (str, 선택적): 그래프의 제목. 기본값은 "distribution".
        ylabel (str, 선택적): y축 레이블. 기본값은 "count".
    """

    pivot_table = data.groupby([main_column, sub_column]).size().unstack()
    
    plt.figure(figsize=(3, 2))

    ax = pivot_table.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Blues')

    for rect in ax.patches:
        height = rect.get_height()
        position = rect.get_y() + height / 2
        ax.text(rect.get_x() + rect.get_width() / 2, position, int(height),
                ha='center', va='center')

    plt.title(title)
    plt.xlabel(str(main_column))
    plt.ylabel(ylabel)
    plt.show()


def text_length(data: pd.DataFrame):
    """
    pandas 데이터프레임의 'instruction'과 'output' 컬럼의 텍스트 길이를 계산하여 데이터프레임 형태로 반환합니다.
    함수는 'instruction' 컬럼의 텍스트를 사용자와 챗봇의 대화로 분리하고, 각 텍스트를 토큰화하여 길이를 계산합니다.
    'output' 컬럼의 텍스트는 챗봇의 대화로 간주하고 동일한 방법으로 길이를 계산합니다.

    Args:
        data (pd.DataFrame): 'instruction'과 'output' 컬럼을 포함하는 데이터프레임.

    Returns:
        user_df (pd.DataFrame): 사용자 텍스트와 그에 대한 토큰화 길이를 포함하는 데이터프레임.
        chatbot_df (pd.DataFrame): 챗봇 텍스트와 그에 대한 토큰화 길이를 포함하는 데이터프레임.
    """
    user = []
    chatbot = []

    for text in data['instruction']:
        sentences = text.split('\n')

        # single turn
        if len(sentences) == 1:
            user.append(sentences[0])

        # multi turn
        elif len(sentences) > 1:
            for sentence in sentences:
                if sentence.startswith("질문:"):
                    user.append(sentence.lstrip('질문: ').rstrip('\n'))
                else:
                    chatbot.append(sentence.lstrip('답변: ').rstrip('\n'))

        else:
            print(f"{text}는 문제가 있군요!", )

    chatbot += data['output'].tolist()

    user_df = pd.DataFrame(user, columns=['user_text'])
    chatbot_df = pd.DataFrame(chatbot, columns=['chatbot_text'])

    tokenizer = AutoTokenizer.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2', use_fast=True)
    tokenize_fn = tokenizer.tokenize

    user_df['length'] = [len(tokenize_fn(text)) for text in user_df['user_text']]
    chatbot_df['length'] = [len(tokenize_fn(text)) for text in chatbot_df['chatbot_text']]
    
    return user_df, chatbot_df


def barplot_binning(series, xlabel, title, bins):
    """
    주어진 series를 막대 그래프로 표시합니다.
    
    Args:
        series (Series): 그래프로 표시할 Series type의 데이터.
        xlabel (str): x축 라벨.
        title (str): 그래프 제목.
        bins (int or sequence): 데이터를 나눌 구간. 정수를 전달하면 해당 수만큼 균등한 구간으로 나눕니다.
    """
    binned_series = pd.cut(series, bins=bins, right=False, include_lowest=True)
    binned_series = binned_series.apply(lambda x: x.left).value_counts().sort_index()
    
    binned_series.index = binned_series.index.astype(str)
    binned_series.index = binned_series.index.where(binned_series.index != binned_series.index[-1], binned_series.index[-1]+' +')  # Append "이상" to the last label
    
    binned_series.plot(kind='bar', color='royalblue')
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(title)
    plt.show()


def count_words(series):
    """
    주어진 텍스트 시리즈에서 각 단어의 출현 빈도를 계산하고, 단어별 빈도수를 내림차순으로 정렬하여 반환하는 함수입니다.
    함수는 Okt 형태소 분석기를 사용하여 텍스트를 단어 단위로 분리하고, 이를 바탕으로 각 단어의 출현 빈도를 계산합니다.

    Args:
        series (pd.Series): 단어 빈도수를 계산할 텍스트 데이터가 포함된 pandas series.

    Returns:
        word_count_df (pd.DataFrame): 각 단어와 그에 대한 빈도수를 포함하는 pandas dataframe.
    """
    okt = Okt()

    word_list = []

    for text in series:
        words = okt.morphs(text)
        word_list.extend(words)
        
    word_count = Counter(word_list)    
    word_count_df = pd.DataFrame(list(word_count.items()), columns=['단어', '빈도수'])
    word_count_df = word_count_df.sort_values(by='빈도수', ascending=False).reset_index(drop=True)
    
    return word_count_df