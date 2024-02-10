from pathlib import Path
from typing import List

import click
from pandas import DataFrame
from progress.bar import Bar
from pypdf import PdfReader


@click.command()
@click.option(
    "pdfFilePath",
    "-i",
    "--input",
    help="Path to PDF (.pdf) file to extract text from",
    type=Path,
    required=True,
)
@click.option(
    "csvFilePath",
    "-o",
    "--output",
    help="Path to CSV (.csv) file to store text in",
    type=Path,
    required=True,
)
def main(pdfFilePath: Path, csvFilePath: Path) -> None:
    stor: dict[str, List] = {"Text": [], "Class": []}

    pdfReader: PdfReader = PdfReader(stream=pdfFilePath)

    pdfPageCount: int = len(pdfReader.pages)

    with Bar("Extracting text from PDF files...", max=pdfPageCount) as bar:
        page: int
        for page in range(pdfPageCount):
            content: List[str] = pdfReader.pages[page].extract_text().split(sep="\n")
            stor["Text"].extend(content)
            bar.next()

    stor["Class"] = [None] * len(stor["Text"])

    df: DataFrame = DataFrame(data=stor)
    df.to_csv(path_or_buf=csvFilePath, index=False)


if __name__ == "__main__":
    main()
