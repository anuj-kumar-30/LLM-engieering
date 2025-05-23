import os
from dotenv import load_dotenv
import requests

load_dotenv(override=True)

notion_token=os.getenv('NOTION_TOKEN')

def get_notion_page_data(page_url, notion_api_key):
    # Extract page ID from URL
    page_id = page_url.split('-')[-1]
    
    # Set up headers with API key
    headers = {
        'Authorization': f'Bearer {notion_api_key}',
        'Notion-Version': '2022-06-28',
        'Content-Type': 'application/json'
    }
    
    # API endpoint for retrieving page content
    url = f'https://api.notion.com/v1/blocks/{page_id}/children'
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        data = response.json()
        
        # Process the blocks into a more usable format
        processed_data = []
        for block in data.get('results', []):
            block_type = block.get('type')
            if block_type == 'paragraph':
                text = ''.join([text.get('plain_text', '') for text in block[block_type].get('rich_text', [])])
                if text:
                    processed_data.append({
                        'type': 'paragraph',
                        'content': text
                    })
            elif block_type == 'heading_1':
                text = ''.join([text.get('plain_text', '') for text in block[block_type].get('rich_text', [])])
                if text:
                    processed_data.append({
                        'type': 'heading_1',
                        'content': text
                    })
            elif block_type == 'heading_2':
                text = ''.join([text.get('plain_text', '') for text in block[block_type].get('rich_text', [])])
                if text:
                    processed_data.append({
                        'type': 'heading_2',
                        'content': text
                    })
            elif block_type == 'bulleted_list_item':
                text = ''.join([text.get('plain_text', '') for text in block[block_type].get('rich_text', [])])
                if text:
                    processed_data.append({
                        'type': 'bullet_point',
                        'content': text
                    })
        
        return processed_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Notion page: {str(e)}")
        return None
    
notion_url = "https://www.notion.so/Machine-Learning-Notes-1dd9a3a6132180b6bccfc7cfa39bd39b?pvs=4"
notion_api_key = notion_token
get_notion_page_data(notion_url, notion_api_key)

'''
# Cell output

Error fetching Notion page: 400 Client Error: Bad Request for url: https://api.notion.com/v1/blocks/1dd9a3a6132180b6bccfc7cfa39bd39b?pvs=4/children


'''