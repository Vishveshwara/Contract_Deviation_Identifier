from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import json
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from typing import Union, List, Tuple, Dict
from langchain_core.agents import AgentFinish


agent_finishes = []

call_number = 0


def fixjson2(badjson):
    # Check if the input JSON is already valid
    try:
        json.loads(badjson)
        return badjson  # Return original JSON if it's valid
    except json.JSONDecodeError:
        pass  # Continue to attempt to fix the JSON if it's invalid

    s = badjson
    idx = 0
    while True:
        try:
            start = s.index('": "', idx) + 4
            end1 = s.find('",\n', idx)
            end2 = s.find('"\n', idx)

            # Handle case where end1 or end2 might not be found correctly
            if end1 == -1:
                end = end2
            elif end2 == -1:
                end = end1
            else:
                end = min(end1, end2)

            if end == -1:  # No more correctable errors found
                break

            content = s[start:end]
            content = content.replace('"', '\\"')
            s = s[:start] + content + s[end:]
            idx = start + len(content) + 6
        except ValueError:
            break  # IndexError or ValueError might occur when indexes are not found

    return s


def print_agent_output(agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish], agent_name: str = 'Generic call'):
    global call_number  # Declare call_number as a global variable
    call_number += 1
    with open("crew_callback_logs.txt", "a") as log_file:
        # Try to parse the output if it is a JSON string
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)  # Attempt to parse the JSON string
            except json.JSONDecodeError:
                pass  # If there's an error, leave agent_output as is

        # Check if the output is a list of tuples as in the first case
        if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
            print(f"-{call_number}----Dict------------------------------------------", file=log_file)
            for action, description in agent_output:
                # Print attributes based on assumed structure
                print(f"Agent Name: {agent_name}", file=log_file)
                print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
                print(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", file=log_file)
                print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
                print(f"Description: {description}", file=log_file)
                print("--------------------------------------------------", file=log_file)

        # Check if the output is a dictionary as in the second case
        elif isinstance(agent_output, AgentFinish):
            print(f"-{call_number}----AgentFinish---------------------------------------", file=log_file)
            print(f"Agent Name: {agent_name}", file=log_file)
            agent_finishes.append(agent_output)
            # Extracting 'output' and 'log' from the nested 'return_values' if they exist
            output = agent_output.return_values
            # log = agent_output.get('log', 'No log available')
            print(f"AgentFinish Output: {output['output']}", file=log_file)
            # print(f"Log: {log}", file=log_file)
            # print(f"AgentFinish: {agent_output}", file=log_file)
            print("--------------------------------------------------", file=log_file)

        # Handle unexpected formats
        else:
            # If the format is unknown, print out the input directly
            print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
            print(type(agent_output), file=log_file)
            print(agent_output, file=log_file)


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        text = parse_pdf(filepath)
        os.environ["GROQ_API_KEY"] = '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'
        llm = ChatGroq(
            # api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",
            temperature=0.5
        )

        class ContractCategorizerAgent():
            def categorizer_agent(self):
                return Agent(
                    role='Contract Categorizer Agent',
                    goal="""This AI assistant, known as the Contract Categorizer Agent, is designed to streamline your contract management process. It will take in a contract's text, created following pre-defined company guidelines, and categorize it into one of the following categories:

                * Employment Contracts: These agreements define the legal relationship between an employer and an employee, outlining responsibilities, compensation, benefits, termination clauses, and other relevant terms.
                * Data Sharing Agreements: These contracts govern the exchange of sensitive or confidential information between two parties, specifying how the data can be used, stored, and protected.
                * Independent Contractor Agreements: These agreements establish a working relationship with an individual or entity who is not considered an employee, outlining the scope of work, payment terms, and ownership of intellectual property.
                * Non-Disclosure Agreements:  These agreements ensure confidentiality of sensitive information shared between parties, restricting its disclosure to unauthorized individuals.
                * Product Sales Agreements: These contracts govern the sale of goods from a seller to a buyer, specifying the product details, purchase price, payment terms, warranties, and delivery conditions.
                * Reseller Agreements:  These agreements authorize a third party (reseller) to sell a company's products to customers, outlining sales territories, pricing, marketing restrictions, and profit-sharing arrangements.
                * Employee Stock Purchase Plan Agreements:  These agreements allow employees to purchase company stock at a discounted price or under specific terms.
                * Severance Agreements: These contracts address the termination of employment, outlining severance pay, non-compete clauses, and confidentiality obligations.
                * Subscription Agreements: These agreements establish a recurring billing relationship for access to a service or product, defining subscription fees, service levels, and termination clauses.

                This categorization helps you efficiently manage and organize your various contractual obligations. """,
                    backstory="""You are a master at understanding what a contract means and are able to categorize it in a useful way.""",
                    llm=llm,
                    verbose=True,
                    allow_delegation=False,
                    max_iter=5,
                    memory=True,
                    step_callback=lambda x: print_agent_output(x, "Contract Categorizer Agent"),
                )

        class ContractCategorizerTasks():
            def categorize_contract(self, contract_content):
                return Task(
                    description=f"""Conduct a comprehensive analysis of the Contract provided and categorize it into one of the predefined categories:

                * Employment Contracts: These agreements define the legal relationship between an employer and an employee, outlining responsibilities, compensation, benefits, termination clauses, and other relevant terms.
                * Data Sharing Agreements: These contracts govern the exchange of sensitive or confidential information between two parties, specifying how the data can be used, stored, and protected.
                * Independent Contractor Agreements: These agreements establish a working relationship with an individual or entity who is not considered an employee, outlining the scope of work, payment terms, and ownership of intellectual property.
                * Non-Disclosure Agreements:  These agreements ensure confidentiality of sensitive information shared between parties, restricting its disclosure to unauthorized individuals.
                * Product Sales Agreements: These contracts govern the sale of goods from a seller to a buyer, specifying the product details, purchase price, payment terms, warranties, and delivery conditions.
                * Reseller Agreements:  These agreements authorize a third party (reseller) to sell a company's products to customers, outlining sales territories, pricing, marketing restrictions, and profit-sharing arrangements.
                * Employee Stock Purchase Plan Agreements:  These agreements allow employees to purchase company stock at a discounted price or under specific terms.
                * Severance Agreements: These contracts address the termination of employment, outlining severance pay, non-compete clauses, and confidentiality obligations.
                * Subscription Agreements: These agreements establish a recurring billing relationship for access to a service or product, defining subscription fees, service levels, and termination clauses.
                * Other Contracts

                    CONTRACT CONTENT:
                    {contract_content}
                    """,
                    expected_output="""A single line giving the category for the type of contract.Most Important **DO NOT print the contract content
                    example: 
                    Non-Disclosure Agreements
                    """,
                    output_file="contract_category.txt",
                    agent=categorizer_agent
                )

        # Instantiate the ContractAgents class and create agents
        ContractCategorizerAgent = ContractCategorizerAgent()
        categorizer_agent = ContractCategorizerAgent.categorizer_agent()
        ContractCategorizerTasks = ContractCategorizerTasks()
        categorize_task = ContractCategorizerTasks.categorize_contract(text)

        crew = Crew(
            agents=[categorizer_agent],
            tasks=[categorize_task],
            process=Process.sequential
        )

        result1 = crew.kickoff()
        print(result1)

        with open(f"contracts\{result1}.txt", 'r') as file:
            contract_template = file.read()

        # Sample Contract Template
        print(contract_template)

        with open(f"ner\{result1}.txt", 'r') as file:
            template = file.read()

        # Key Entity Recognizer Template
        print(template)

        class ContractAgents():
            def contract_KeyEntity_Recognizer(self):
                return Agent(
                    role='Key Entity Scraper',
                    goal="""This AI assistant, known as the Contract Key Entity Recognizer Agent, is designed to streamline the process of extracting key information from the contract. It will analyze the contract text and identify essential details specific to the type of the contract.
                    and the output should be the matching key entities as well as the deviations from the contract(the entities not present) using the template which is mentioned in the task and should strictly follow it.
                    for example if the template has effective date in the template make sure to identify the date and if it is not present mention it as a deviation
                    """,
                    backstory="""You are a master at understanding at referring to templates and finding the key entities which are required and which are deviated and return the matched and deviated entites""",
                    llm=llm,
                    verbose=True,
                    max_iter=20,
                    allow_delegation=False,
                    memory=True,
                    step_callback=lambda x: print_agent_output(x, "Key Entity Scraper"),
                )

            def contract_deviation_agent(self):
                return Agent(
                    role='Deviations Identifier',
                    goal="""This AI assistant, known as the Contract Deviation Agent, is designed to streamline the process of Identifying, reporting and explaining deviations from a template for a given contract. It will analyze the contract text and identify, report and explain deviations(in clauses, sub-clauses or finer text) with respect to the given template.
                      and the output is the list of deviations of the contract from the template along with an explanation for each(how they differ).This agent should not pause and do this task thoroughly without making mistakes or pausing or stopped randomly.
                      Expected Output:
                      A set of bullet points of Deviations found after comparing the contract content with the template \
                    and clear explanations of why and how these deviations are found, and if no deviations present give output as No Deviations are present
                    
                    this is an example template of how the output should be given: 
                    
                    * The template has X clauses, while the contract content has Y clauses. This is a deviation from the template.
                    * Clause N in the template mentions "SOME SPECIFIC TERM", but this term is not present in the contract content. This is a deviation from the template.
                    * Clause N in the template defines "ANOTHER SPECIFIC TERM", but this definition is not present in the contract content. This is a deviation from the template.
                    * Clause N in the template mentions several key points, such as "KEY POINT 1", "KEY POINT 2", and "KEY POINT 3", but these are not present in the contract content. This is a deviation from the template.
                    * Clause N in the template defines "YET ANOTHER SPECIFIC TERM", but this definition is not present in the contract content. This is a deviation from the template.
                
                    *Explanations:*
                
                    * The contract content does not follow the exact structure and clauses of the template.
                    * The definitions and clauses mentioned in the template are not present in the contract content.
                    * The contract content has a different scope and purpose compared to the template.
                    
                    
                    *Conclusion:*
                    
                    The contract content deviates significantly from the provided template. The deviations include differences in structure, definitions, and clauses. A thorough review of the contract content is necessary to ensure that it meets the intended purposes and complies with relevant laws and regulations.

                      
                      """,
                    backstory="""You are a master at understanding at referring to contract templates and finding the deviations and return the deviated clauses, sub clauses and text along with how and why they deviate""",
                    llm=llm,
                    verbose=True,
                    max_iter=50,
                    allow_delegation=False,
                    memory=True,
                    step_callback=lambda x: print_agent_output(x, "Deviations Identifier"),
                )

            def highlighter_agent(self):
                return Agent(
                    role='Highlighter Agent',
                    goal="""This AI assistant, known as the Highlighter Agent, is designed to highlight deviations in the contract and provide explanations. It will identify the parts of the contract that deviate from the template and explain the significance of each deviation.
                    the output should be in the format of a json file,
                    do not use double quotes in the text and explanation as the json format won't work
                    for example:
                    [
                    {
                        "text": "Disclosing Party: ABC Corporation, located at 123 Business Rd, Business City, BS12345",
                        "explanation": "Disclosing Party refers to the entity sharing confidential information."
                    },
                    {
                        "text": "Receiving Party: XYZ Solutions, located at 789 Innovation St, Tech City, TC98765",
                        "explanation": "Receiving Party refers to the entity receiving the confidential information."
                    }
                    ]
                    DO NOT USE Double Quotes in 
                    """,
                    backstory="""You are a detail-oriented highlighter, focusing on deviations in contracts and providing clear explanations for each highlighted deviation.""",
                    llm=llm,
                    verbose=True,
                    max_iter=20,
                    allow_delegation=False,
                    memory=True,
                    step_callback=lambda x: print_agent_output(x, "Highlighter Agent"),
                )

            def json_rectifier(self):
                return Agent(
                    role='Json Rectifier',
                    goal='Ensure JSON strings are properly formatted and valid.',
                    verbose=True,
                    max_iter=20,
                    allow_delegation=False,
                    memory=True,
                    llm=llm,
                    step_callback=lambda x: print_agent_output(x, "Json Rectifier Agent"),
                    backstory=(
                        "You have a keen eye for detail and excel in ensuring data is properly structured."
                        " Your job is to verify that JSON strings adhere to all formatting rules."
                        "Smartly identify if there are any errors and dont include more errors Please"
                    ))

        class ContractTasks():
            def contract_KeyEntity_Recogniser(self, contract_content, contract_type, template):
                return Task(
                    description=f"""Conduct a comprehensive analysis of the contract provided and the category is {contract_type}\
                    and using the template below find the deviations and key entities found in the contract
                    use the template below to find the key entities:
                    {template}

                    use the template to find the key entities by going through every single key entity given in the template as well as using your knowledge to find the key entities and deviations.
                    Every single heading and subheading needs to be matched with the contract given to gather the most details, and use your vast knowledge to identify other key entities like location,parties and make the key entity recognition more robust
                    CONTRACT CONTENT:\n\n {contract_content} \n\n
                    """,
                    expected_output="""A set of bullet points of Key Entities found with meaning and relationship with Parties \
                    and clear bullet points of deviations or key entities or relationships which are not found and if no deviations present give output as Required Entites are present
                    
                    Example output:
                    **Key Entities:**

                    * **Parties:**
                        + #PARTY A NAME#: [whose principal place of residence is at / a #PARTY A JURISDICTION# corporation with its principal place of business at #PARTY A ADDRESS#] (the "#PARTY A#")
                        + #PARTY B NAME#: [whose principal place of residence is at / a #PARTY B JURISDICTION# corporation with its principal place of business at #PARTY B ADDRESS#] (the "#PARTY B#")
                    * **Data:**
                        + "Data" includes #DESCRIPTION OF THE DATA#, further described in #ATTACHMENT#, attached to this agreement.
                    * **Effective Date:** the "Effective Date" is the last date signed by the parties.
                    * **Purpose:** the parties are entering into this agreement for the purpose of #INSERT SHORT DESCRIPTION OF PURPOSE OF THE DATA USE# (the "Purpose").
                    * **Term:** the agreement will commence on the Effective Date and continue as long as #PARTY B# retains the Data, unless terminated earlier (the "Term").
                    * **License Grant:** #PARTY A# hereby grants to #PARTY B# a limited, non-exclusive, non-transferable, and revocable license to access, copy, and use the Data (the "#DELIVERABLES#").
                    
                    Key Entities Deviations or Relationships which are not found:
                    
                    * **Data Security and Confidentiality:** There is no specific mention of data security and confidentiality measures in the contract.
                    * **Data Handling:** There is no specific mention of data handling procedures in the contract.
                    * **Data Sharing:** The contract does not specify the specific data sharing agreements or protocols.
                    * **Intellectual Property:** The contract does not specify intellectual property rights or ownership.
                    * **Governing Law and Dispute Resolution:** The contract does not specify the governing law or dispute resolution mechanisms.

                    
                    """,
                    context=[],
                    output_file=f"contract_ner_info.txt",
                    agent=contract_KeyEntity_Recognizer
                )

            '''def key_entity_re_verifier(self, contract_content: str, contract_type: str, template: str) -> Task:
                return Task(
                    description=f"""Re-check the contract for the presence of all key entities and deviations identified previously. The category is {contract_type}.
                    Use the template below to re-verify the key entities:
                    {template}
                    any key entities from the template which is missing in the recognized key entities then re-check and confirm if the key entities are actually present or not.
                    do not trust the output from the key_entity_re_verifier_agent and perform verification yourself in a professional manner

                    CONTRACT CONTENT:\n\n {contract_content} \n\n
                    """,
                    expected_output="""A set of bullet points of Key Entities found with meaning and relationship with Parties \
                    and clear bullet points of deviations or key entities or relationships which are not found and if no deviations present give output as Required Entites are present""",
                    context=[contract_KeyEntityTask],
                    output_file="ner_verification_report.txt",
                    agent=key_entity_re_verifier_agent
                )'''

            def contract_Deviation_Identifier(self, contract_content, contract_type, contract_template):
                return Task(
                    description=f"""Should not pause and start the task quickly,Conduct a comprehensive analysis of the contract provided and the category is {contract_type}\
                    and using the template below find the deviations in the contract content.
                    use the template below to compare and find the deviations following the format in the expected_output:
                    
                    {contract_template}

                    print the template and use the template to find the deviations by going through every line, clause and sub-clause given in the template as well as the contract content to find every way in which the contract content deviates from the template.
                    Every single heading and subheading needs to be matched with the contract given to gather the most details, and use your vast knowledge to identify other deviations and make the deviation recognition more robust
                    CONTRACT CONTENT:\n\n {contract_content} \n\n
                    
                    Please follow the expected output format and mention clause numbers and give clear explanations.
                
                    """,
                    expected_output="""A set of bullet points of Deviations found after comparing the contract content with the template \
                    and clear explanations of why and how these deviations are found, and if no deviations present give output as No Deviations are present
                    
                    this is an example template of how the output should be given: 
                    
                    * The template has X clauses, while the contract content has Y clauses. This is a deviation from the template.
                    * Clause N in the template mentions "SOME SPECIFIC TERM", but this term is not present in the contract content. This is a deviation from the template.
                    * Clause N in the template defines "ANOTHER SPECIFIC TERM", but this definition is not present in the contract content. This is a deviation from the template.
                    * Clause N in the template mentions several key points, such as "KEY POINT 1", "KEY POINT 2", and "KEY POINT 3", but these are not present in the contract content. This is a deviation from the template.
                    * Clause N in the template defines "YET ANOTHER SPECIFIC TERM", but this definition is not present in the contract content. This is a deviation from the template.
                
                    *Explanations:*
                
                    * The contract content does not follow the exact structure and clauses of the template.
                    * The definitions and clauses mentioned in the template are not present in the contract content.
                    * The contract content has a different scope and purpose compared to the template.
                    
                    
                    *Conclusion:*
                    
                    The contract content deviates significantly from the provided template. The deviations include differences in structure, definitions, and clauses. A thorough review of the contract content is necessary to ensure that it meets the intended purposes and complies with relevant laws and regulations.

                    """,
                    context=[],
                    output_file=f"contract_info.txt",
                    agent=contract_deviation_agent
                )

            def highlighter_task(self, contract_content, contract_type, contract_template):
                return Task(
                    description=f"""Highlight the deviations in the contract content provided and provide explanations. The deviations is passed as context,
                    from the contract deviation identifier task. The category is {contract_type}.

                    MOST IMPORTANT RULE: 
                    only highlight the text from the {contract_content} and please give in the format given in expected output.
                    and it must follow json standards for the output.

                    CONTRACT CONTENT:\n\n {contract_content} \n\n
                    """,
                    expected_output="""A list of highlighted deviations in the format: And do not add any other text and give it in json form like given below, do not give any note statements
                    
                    Example Output: 
                    [
                        {
                            "text": "Any sentence or paragraph from contract_content",
                            "explanation": "Explanation of the deviation here"
                        },
                        {
                        "text": "This Non-Disclosure Agreement ("Agreement") is made and entered into as of July 1, 2024 ("Effective Date"), by and between:",
                        "explanation": "The template has 38 clauses, while the contract content has 9 clauses. This is a deviation from the template."
                    }
                    ]
                    """,
                    context=[contract_Deviation_IdentifierTask],
                    output_file="highlighted_deviations.txt",
                    agent=highlighter_agent
                )

            def jsonverify(self):
                return Task(
                    description=
                    """Read the provided .txt file, convert its content to a JSON string, and validate it.Ensure the JSON has the correct format, with proper use of commas, quotes, and no extra spaces.Return 'Valid JSON' if the string is correctly formatted, otherwise return the specific error.
                    dont include any other text other than the json format.
                            example :[
                        {
                            "text": "Any sentence or paragraph from contract_content",
                            "explanation":          "Explanation of the deviation here"
                        },
                        {
                        "text": "This Non-Disclosure Agreement ("Agreement") is made and entered into as of July 1, 2024 ("Effective Date"), by and between:",
                        "explanation": "The template has 38 clauses, while the contract content has 9 clauses. This is a deviation from the template."
                    }
                    ] should be converted to 
                     [
                        {
                            "text": "Any sentence or paragraph from contract_content",
                            "explanation": "Explanation of the deviation here"
                        },
                        {
                        "text": "This Non-Disclosure Agreement (\"Agreement\") is made and entered into as of July 1, 2024 (\"Effective Date\"), by and between:",
                        "explanation": "The template has 38 clauses, while the contract content has 9 clauses. This is a deviation from the template."
                    }
                    ]
                    removing the extra space and adding \ before quotes """,
                    expected_output='A string indicating whether the JSON is valid or an error message.',
                    agent=json_rectifier_agent,
                    context=[highlight_task],
                    output_file="repaired_highlighted_deviations.json",
                )

        agents = ContractAgents()
        tasks = ContractTasks()

        contract_KeyEntity_Recognizer = agents.contract_KeyEntity_Recognizer()
        contract_deviation_agent = agents.contract_deviation_agent()
        highlighter_agent = agents.highlighter_agent()
        json_rectifier_agent = agents.json_rectifier()

        contract_KeyEntityTask = tasks.contract_KeyEntity_Recogniser(contract_content=text,
                                                                     contract_type=result1, template=template)
        contract_Deviation_IdentifierTask = tasks.contract_Deviation_Identifier(text, result1,
                                                                                contract_template)
        highlight_task = tasks.highlighter_task(text, result1, contract_template)
        json_rectify = tasks.jsonverify()

        crew = Crew(
            agents=[contract_KeyEntity_Recognizer, contract_deviation_agent, highlighter_agent,json_rectifier_agent],
            tasks=[contract_KeyEntityTask, contract_Deviation_IdentifierTask, highlight_task,json_rectify],
            process=Process.sequential
        )

        # Kick off the process
        result2 = crew.kickoff()
        print(result2)

        try:
            f = open('repaired_highlighted_deviations.json')
        except:
            try:
                # Open the file in read mode ('r')
                with open("highlighted_deviations.txt", 'r') as file:
                    # Read the entire content of the file into a variable
                    badjson = file.read()
                    goodjson = fixjson2(badjson)
                    json_data = json.loads(goodjson, strict=False)
                    with open('repaired_highlighted_deviations.json', 'w') as outfile:
                        json.dump(json_data, outfile, indent=4)
            except Exception as e:
                print("Error:{e}")

        f = open('repaired_highlighted_deviations.json')
        highlights = json.load(f)

        with open(f"contract_info.txt", 'r') as file:
            result2 = file.read()

        with open(f"contract_ner_info.txt", 'r') as file:
            key_entity = file.read()

        response = {
            "parsed_text": text,
            "contract_category": result1,
            "contract_template": contract_template,
            "key_entity_template": template,
            "key_entity": key_entity,
            "final_output": result2,
            'highlights': highlights
        }
        return jsonify(response)


def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text


if __name__ == '__main__':
    app.run(debug=True)
