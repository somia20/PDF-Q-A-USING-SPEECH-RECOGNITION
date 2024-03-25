# main_script.py
from QA import answer_question


def main():
    # Specify the path to your PDF file
    pdf_path = "c:/Users/User/Desktop/AI.pdf"

    # Example question
    question = "AI Vs Robotics?"

    # Call answer_question function
    answer = answer_question(pdf_path, question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
