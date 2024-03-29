from functools import partial
from pathlib import Path

import marvin
import pandas as pd
from dreamai_gen.asking import ask
from dreamai_gen.llms import ModelName, ask_oai
from dreamai_gen.utils import deindent
from pydantic import BaseModel, Field
from tenacity import RetryError, Retrying, stop_after_attempt
from termcolor import colored

ATTEMPTS = 3
MODEL_NAME = ModelName.GPT_3
MODEL_TEMPERATURE = 0.3
COURSE_NAME = "101"
COURSES_FOLDER = "courses"
COMPARISONS_NAME = "comparisons"

marvin.settings.openai.chat.completions.model = MODEL_NAME
marvin.settings.openai.chat.completions.temperature = MODEL_TEMPERATURE


class Comparison(BaseModel):
    topics_from_the_course_outline_covered_in_the_book: list[str] = Field(
        ...,
        description=deindent(
            """
            List of topics from the course outline that are covered in the book.
            For each topic, give me the chapter number if you can find it.
            """
        ),
    )
    topics_from_the_course_outline_not_covered_in_the_book: list[str] = Field(
        ...,
        description="List of topics from the course outline that are not covered in the book",
    )


def gen_comparisons(
    course_name: str = COURSE_NAME,
    courses_folder: str | Path = COURSES_FOLDER,
    attempts: int = ATTEMPTS,
    oai_model: str = MODEL_NAME,
) -> dict:
    course_folder = Path(courses_folder) / course_name
    course_prompt = course_folder / f"{course_name}_prompt.txt"
    course_toc_folder = course_folder / "toc"
    course_comaprisons_file = course_folder / f"{course_name}_{COMPARISONS_NAME}.xlsx"
    compared_books = []
    if course_comaprisons_file.exists():
        comparison_df = pd.read_excel(course_comaprisons_file)
        compared_books = comparison_df["Book"].unique().tolist()
    comparisons = {}
    asker = partial(ask_oai, model=oai_model)
    for course_toc in course_toc_folder.glob("*.txt"):
        if course_toc.stem in compared_books:
            continue
        try:
            for attempt in Retrying(stop=stop_after_attempt(attempts)):
                with attempt:
                    comparison = ask(
                        course_prompt,
                        course_toc,
                        {"data": asker},
                        {"comparison": partial(marvin.cast, target=Comparison)},
                    ).get("comparison")
        except RetryError:
            print(
                colored(
                    f"Failed to compare {course_toc.stem} after {attempts} attempts.",
                    "red",
                )
            )
            continue
        if comparison:
            comparison = comparison.model_dump()
            max_len = max([len(v) for v in comparison.values()])
            comparison = {
                k.replace("_", " ").title(): v + [""] * (max_len - len(v))
                for k, v in comparison.items()
            }
            comparison["Book"] = course_toc.stem
            comparisons[course_toc.stem] = comparison
    return comparisons


def gen_comparisons_excel(
    course_name: str = COURSE_NAME, courses_folder: str | Path = COURSES_FOLDER
) -> pd.DataFrame:
    comparisons = gen_comparisons(
        course_name=course_name, courses_folder=courses_folder
    )
    if not comparisons:
        return pd.DataFrame()
    excel_file = pd.concat(
        [pd.DataFrame(comparison) for comparison in comparisons.values()]
    )
    course_folder = Path(courses_folder) / course_name
    course_comaprisons_file = course_folder / f"{course_name}_{COMPARISONS_NAME}.xlsx"
    if course_comaprisons_file.exists():
        old_excel_file = pd.read_excel(course_comaprisons_file)
        excel_file = pd.concat([old_excel_file, excel_file])
    excel_file.to_excel(course_comaprisons_file, index=False)
    return excel_file
