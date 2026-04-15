import asyncio
import httpx
import time

CHAT_URL = "http://localhost:8000/chat/"
SECTION_URL = "http://localhost:8000/section/"

async def send_chat_request(client, i):
    start = time.time()

    resp = await client.post(CHAT_URL, json={
        "query": "What is constitution of india in 1 sentence?"
    })

    elapsed = time.time() - start

    try:
        data = resp.json()
    except:
        data = {}

    print(f"\n[CHAT] {'='*50}")
    print(f"Request  : {i}")
    print(f"Status   : {resp.status_code}")
    print(f"Time     : {elapsed:.2f}s")
    print(f"Answer   : {data.get('answer', '')[:100]}...")

async def send_section_request(client, i):
    start = time.time()

    resp = await client.post(SECTION_URL, json={
        "query": "Section 3 of information technology act"
    })

    elapsed = time.time() - start

    try:
        data = resp.json()
    except:
        data = {}

    print(f"\n[SECTION] {'='*50}")
    print(f"Request  : {i}")
    print(f"Status   : {resp.status_code}")
    print(f"Time     : {elapsed:.2f}s")
    print(f"Answer   : {data.get('answer', '')[:300]}...")

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        
        tasks = []

        # 10 chat requests
        tasks += [send_chat_request(client, i) for i in range(10)]

        # 10 section requests
        tasks += [send_section_request(client, i) for i in range(10)]

        await asyncio.gather(*tasks)

asyncio.run(main())