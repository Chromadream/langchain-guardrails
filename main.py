from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
from chatrailchain import ChatRAILChain

load_dotenv()

def main():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    query = """Given the following doctor's notes about a patient,
please extract a dictionary that contains the patient's information.  

{{doctors_notes}}"""
    rail_spec = """
<object name="patient_info">
    <string name="gender" description="Patient's gender" />
    <integer name="age" format="valid-range: 0 100"/>  
    <list name="symptoms" description="Symptoms that the patient is currently experiencing. Each symptom should be classified into  separate item in the list.">
        <object>
            <string name="symptom" description="Symptom that a patient is experiencing" />
            <string name="affected area" description="What part of the body the symptom is affecting"
                format="valid-choices: {['head', 'neck', 'chest']}"
                on-fail-valid-choices="reask"
            />  
        </object>
    </list>
    <list name="current_meds" description="Medications the patient is currently taking and their response">
        <object>
            <string name="medication" description="Name of the medication the patient is taking" />
            <string name="response" description="How the patient is responding to the medication" />
        </object>
    </list>
</object>
"""
    chain = ChatRAILChain(llm=llm, query=query, rail_output_spec=rail_spec)
    response = chain.run({'doctors_notes': """49 y/o Male with chronic macular rash to face & hair, worse in beard, eyebrows & nares.
Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream"""}, callbacks=[StdOutCallbackHandler()])
    print(response)
    

if __name__ == "__main__":
    main()