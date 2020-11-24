import asyncio

async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')

async def say_after(s, t):
    await asyncio.sleep(t)
    print(s)

asyncio.run(main())
