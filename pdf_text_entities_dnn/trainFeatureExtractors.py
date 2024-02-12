from pathlib import Path
from typing import List, Literal, Tuple

import click
import pandas
from pandas import DataFrame
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)


def readCSV(csvFilePath: Path) -> DataFrame:
    return pandas.read_csv(filepath_or_buffer=csvFilePath)


def generateCorpus(df: DataFrame) -> List[str]:
    return df["Text"].to_list()


def trainCountVectorizer(
    corpus: List[str],
    stopWords: Literal["english"] | None = "english",
    ngramRange: Tuple[int, int] = (1, 2),
    analyzer: Literal["word", "char_wb"] = "char_wb",
) -> CountVectorizer:
    cv: CountVectorizer = CountVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=stopWords,
        ngram_range=ngramRange,
        analyzer=analyzer,
    )
    cv.fit(raw_documents=corpus)
    return cv


def trainHashingVectorizer(
    corpus: List[str],
    stopWords: Literal["english"] | None = "english",
    ngramRange: Tuple[int, int] = (1, 2),
    analyzer: Literal["word", "char_wb"] = "char_wb",
) -> HashingVectorizer:
    hv: HashingVectorizer = HashingVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=stopWords,
        ngram_range=ngramRange,
        analyzer=analyzer,
    )
    hv.fit(X=corpus)
    return hv


def trainTFIDFVectorizer(
    corpus: List[str],
    stopWords: Literal["english"] | None = "english",
    ngramRange: Tuple[int, int] = (1, 2),
    analyzer: Literal["word", "char_wb"] = "char_wb",
) -> TfidfVectorizer:
    tfidf: TfidfVectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=stopWords,
        ngram_range=ngramRange,
        analyzer=analyzer,
    )
    tfidf.fit(raw_documents=corpus)
    return tfidf


@click.command()
@click.option(
    "csvFilePath",
    "-i",
    "--input",
    help="Path to CSV file containing text and classes",
    required=True,
    type=Path,
)
def main(csvFilePath: Path) -> None:
    df: DataFrame = readCSV(csvFilePath=csvFilePath).dropna(ignore_index=True)
    corpus: List[str] = generateCorpus(df=df)


if __name__ == "__main__":
    main()
