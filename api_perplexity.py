import time, re, asyncio
from playwright.sync_api import sync_playwright, Playwright
from playwright.async_api import async_playwright
from langchain_core.messages import AIMessage
from bs4 import BeautifulSoup

class table_building_LLM():
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = "llama-3-70b-instruct"
    
    def send_request(self, playwright: Playwright, table_data, file_type, task):
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
    
        page.goto("https://labs.perplexity.ai/")
        page.locator('#lamma-select').select_option(self.model)
    
        item_statement = "{" + "itemize" + "}"
    
        if file_type == 'latex':
            file_format = 'LaTeX'
        elif file_type == 'md':
            file_format = 'Markdown'
    
        if task == 'summary':
            format = '''Here's the table information:
                    \\begin{itemize}\n
                    \\item [Row summary for the first row]\n
                    \\item [Row summary for the second row]\n
                    ...\n
                    \\end{itemize}\n
                '''
            question = f'You are a power systems expert with proficiency in {file_format}. Given the {file_format} table snippet provided below:\n{table_data}, please analyze the data and summarize the information contained in each row concisely. Format your summary as an unnumbered list in LaTeX syntax using the following structure:{format} Ensure that your response includes one \item for each row in the table snippet, with each item beginning with \\item. The response should be contained within the \\begin{item_statement} and \\end{item_statement} commands. Make sure the tabular data is described accurately'
    
        elif task == 'qa':
            format = '''Here's some question answer pairs:
                    \\begin{itemize}\n
                    \\item Question:(question 1) Answer:(answer to question 1)\n
                    \\item Question:(question 2) Answer:(answer to question 2)\n
                    ...\n
                    \\end{itemize}\n
                '''
            question = f'"You are an expert in power systems with a knack for creating detailed Questions and Answers from textual data. Given the {file_format} file snippet provided below:\n{table_data}, craft a series of very descriptive questions and answers. Each question should provide all the available contextual information about the topic/entity being to avoid vagueness. Format your response as an unnumbered list in LaTeX syntax using the following structure:{format}. Start each entry with \item, contained within \\begin{item_statement} and \\end{item_statement}. Ensure each question answer pair in your response corresponds to one \item in your list. Make as many descriptive question answer pairs as possible"'
            
        page.locator('textarea[placeholder="Ask anything..."]').fill(question)
    
        page.locator('button:has(svg.fa-arrow-up)').click()
    
        # Pulls the page each second and sees if LLM has finished printing text
        prev_time_text = ""
        same_count = 0
        for counter in range(self.sleep_time):
            # Grab the time stamp from the hotbar
            html_content = page.content()
            current_soup = BeautifulSoup(html_content, 'html.parser')
            hotbar_html = current_soup.find('div', class_='pl-md')
            if hotbar_html:
                #Checks if the hotbar text has changed(not the actual time value)
                time_text = hotbar_html.text.strip()
                if time_text == prev_time_text:
                    if same_count > 2:
                        break
                    else:
                        same_count = same_count + 1
                else:
                    same_count = 0
                    prev_time_text = time_text
                    
            # Sleep for 1 sec until the specified sleep time is reached
            time.sleep(1)
    
        html_content = page.content()
    
        soup = BeautifulSoup(html_content, 'html.parser')
        items = soup.find_all(text=lambda text: text and "\item" in text)
    
        new = ''
        for item in items[1:]:
            new += item
            new += '\n'
    
        context.close()
        browser.close()
    
        return new
    
    def get_tabular_data(self, table_data, file_type, task):
        with sync_playwright() as playwright:
            response = send_request(playwright, table_data, file_type, task)
        return response


# For executing general QnA with Perplexity Lab
class fake_LLM():
    def __init__(self, model=None, sleep_time=None, verbose=None):
        if model is not None:
            self.model = model
        else:
            self.model = "llama-3-70b-instruct"
        if sleep_time is not None:
            self.sleep_time = sleep_time
        else:
            self.sleep_time = 15
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = False

    def invoke(self, question):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False) #Must have brower GUI enabled for this to work
            context = browser.new_context()
            page = context.new_page()
        
            page.goto("https://labs.perplexity.ai/")
            page.locator('#lamma-select').select_option(self.model)
            
            page.locator('textarea[placeholder="Ask anything..."]').fill(question)
            page.locator('button:has(svg.fa-arrow-up)').click()

            # Pulls the page each second and sees if LLM has finished printing text
            prev_time_text = ""
            same_count = 0
            for counter in range(self.sleep_time):
                # Grab the time stamp from the hotbar
                html_content = page.content()
                current_soup = BeautifulSoup(html_content, 'html.parser')
                hotbar_html = current_soup.find('div', class_='pl-md')
                if hotbar_html:
                    #Checks if the hotbar text has changed(not the actual time value)
                    time_text = hotbar_html.text.strip()
                    if time_text == prev_time_text:
                        if same_count > 2:
                            break
                        else:
                            same_count = same_count + 1
                    else:
                        same_count = 0
                        prev_time_text = time_text
                        
                # Sleep for 1 sec until the specified sleep time is reached
                time.sleep(1)
            
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            body = soup.find('body')
            body_text = body.get_text()

            start_index = 0
            end_index = 0
            all_matches = []
            for i in range(len(body_text)):
                char = body_text[i]
                if "LLM served by Perplexity Labs" in body_text[i:i+29]:
                    start_index = i + 29
                if "CopyAsk Perplexity" in body_text[i:i+18]:
                    end_index = i
                    all_matches.append(body_text[start_index:end_index])

            try:
                response = all_matches[-1]
            except:
                print("The LLM response could not be found(likely because the sleep window was to short for it to respond).")
                response = ""
            
            context.close()
            browser.close()
            if self.verbose == True:
                print(response)
            
            return AIMessage(content=response,id="labs.perplexity.ai",name=self.model)

# For executing general QnA with Perplexity Lab
class async_fake_LLM():
    def __init__(self, model=None, sleep_time=None, verbose=None):
        self.TIMEOUT_VALUE = 3 # Number of times it will retry invoking the LLM if no response is returned
        if model is not None:
            self.model = model
        else:
            self.model = "llama-3-70b-instruct"
        if sleep_time is not None:
            self.sleep_time = sleep_time
        else:
            self.sleep_time = 15
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = False

    async def invoke(self, question):
        response = ""
        for attempt in range(self.TIMEOUT_VALUE):
            response = await self.send_request(question)
            if response != "LLM_ERROR":
                break

        if attempt == self.TIMEOUT_VALUE:
            print(f"Failed to contact Perplexity LLM after {attempt} attempts")
        
        return response

    async def send_request(self, question):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=False) #Must have brower GUI enabled for this to work
            context = await browser.new_context()
            page = await context.new_page()
        
            await page.goto("https://labs.perplexity.ai/")
            await page.locator('#lamma-select').select_option(self.model)
            await page.locator('textarea[placeholder="Ask anything..."]').fill(question)
            await page.locator('button:has(svg.fa-arrow-up)').click()

            # Pulls the page each second and sees if LLM has finished printing text
            prev_time_text = ""
            same_count = 0
            for counter in range(self.sleep_time):
                # Grab the time stamp from the hotbar
                html_content = await page.content()
                current_soup = BeautifulSoup(html_content, 'html.parser')
                hotbar_html = current_soup.find('div', class_='pl-md')
                if hotbar_html:
                    #Checks if the hotbar text has changed(not the actual time value)
                    time_text = hotbar_html.text.strip()
                    if time_text == prev_time_text:
                        if same_count > 2:
                            break
                        else:
                            same_count = same_count + 1
                    else:
                        same_count = 0
                        prev_time_text = time_text
                        
                # Sleep for 1 sec until the specified sleep time is reached
                time.sleep(1)
            
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            body = soup.find('body')
            body_text = body.get_text()

            start_index = 0
            end_index = 0
            all_matches = []
            for i in range(len(body_text)):
                char = body_text[i]
                if "LLM served by Perplexity Labs" in body_text[i:i+29]:
                    start_index = i + 29
                if "CopyAsk Perplexity" in body_text[i:i+18]:
                    end_index = i
                    all_matches.append(body_text[start_index:end_index])

            try:
                response = all_matches[-1]
            except:
                print("The LLM response could not be found(likely because the sleep window was to short for it to respond).")
                response = "LLM_ERROR"
            
            await context.close()
            await browser.close()
            if self.verbose == True:
                print(response)
            
            return AIMessage(content=response,id="labs.perplexity.ai",name=self.model)