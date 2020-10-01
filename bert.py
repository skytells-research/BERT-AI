from transformers import pipeline

class QA: 
    def __init__(self):
        return self

    def getAnswer(question, context):
        question_answerer = pipeline('question-answering')
        out = question_answerer({'question': question,'context': context})
        #print(out)
        return out

    def generateText(context, max_length, do_sample=False):
        text_generator = pipeline("text-generation")
        resutls = text_generator(context, max_length=max_length, do_sample=do_sample)
        print(resutls)
        return resutls