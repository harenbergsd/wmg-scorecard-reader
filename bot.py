import os
import discord
from dotenv import load_dotenv
from discord.ext import commands
from scorecard import Scorecard
from PIL import Image
import aiohttp
from io import BytesIO
from table2ascii import table2ascii as t2a, PresetStyle
import hashlib
import asyncio
import threading

import misc

file_lock = threading.Lock()

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix="!", intents=intents)
scorecard_cache = {}


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


@bot.group(invoke_without_command=True)
async def larry(ctx):
    await ctx.send("Usage: `!larry <command> [option(s)] [message_link(s)]`\nType `!larry help` for details.")


@larry.command(name="help")
async def help_command(ctx):
    embed = discord.Embed(
        title="LARRY Bot Help",
        description="Reviews Walkabout Minigolf scorecards.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Commands",
        value="""
        **help** - Show this help message        
        **review** - Reviews one or more scorecards
        """,
        inline=False,
    )
    embed.add_field(
        name="Usage",
        value="Upload image(s) and, in the same message, type:\n"
        "```!larry review```\n"
        "Or provide a (or multiple) Discord message link(s) with images:\n"
        "```!larry review <message-link1> ... <message-linkN>```",
        inline=False,
    )
    embed.set_footer(text="Larry: Lifeless Algorithm Rapidly Reviewing Your scorecard")
    await ctx.send(embed=embed)


def df_to_str(df):
    header = [df.index.name or ""] + list(df.columns)
    body = [[idx] + list(row) for idx, row in zip(df.index, df.values)]

    table_str = t2a(
        header=header,
        body=body,
        style=PresetStyle.thin_compact,
    )

    return table_str


def split_scorecard(scorecard):
    df = scorecard.df
    total = df["total"]
    df = df.drop(columns=["total"], errors="ignore")

    # Split the scorecard into two halves
    front_nine = df.iloc[:, :9]
    back_nine = df.iloc[:, 9:]

    # Add total columns to both
    front_nine["out"] = front_nine.sum(axis=1)
    back_nine["in"] = back_nine.sum(axis=1)
    back_nine["tot"] = total

    return front_nine, back_nine


async def process_images(images, ctx):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _process_images_sync, images, ctx, loop)


def _process_images_sync(images, ctx, loop):
    with file_lock:
        standard_contours = misc.load_standard_contours()
        scorecard = None
        for img in images:
            hashval = hash_bytes(img)
            if hashval in scorecard_cache:
                s = scorecard_cache[hashval]
            else:
                s = Scorecard.from_image(img, standard_contours)
                scorecard_cache[hashval] = s
            scorecard = s.copy() if scorecard is None else scorecard.combine(s)

        sc = scorecard.copy()
        sc.include_pars = True
        sc.include_best = True
        sc = sc.sorted_by_total()
        if sc is not None:
            text = sc.course
            df = sc.summarize_scores()
            text += f"\n```{df_to_str(df)}```"

            df = sc.summarize_shots()
            text += f"```{df_to_str(df)}```"

            par_comp = sc.compare_to_par().df
            best_comp = sc.compare_to_best().df
            titles = ["Ordered Scorecard", "Scores Compared to Par", "Scores Compared to Best Hole Score"]
            misc.dfs_to_image([sc.df, par_comp, best_comp], titles=titles, output_path="__tmp_tables.png")

            # csv file
            b = BytesIO()
            scorecard.df.to_csv(b)
            b.seek(0)
            csv_file = discord.File(b, filename="scorecard.csv")

            asyncio.run_coroutine_threadsafe(_send_result(ctx, text, csv_file), loop)


async def _send_result(ctx, text, csv_file=None):
    with open("__tmp_tables.png", "rb") as f:
        files = [discord.File(f)]
        if csv_file:
            files.append(csv_file)
        await ctx.send(text, files=files)


@larry.command(name="review")
async def review_command(ctx, *args):
    image_urls = []

    # Check if there are attachments in the current message
    for attachment in ctx.message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            image_urls.append(attachment.url)

    # If user passed message links, try to fetch images from those
    if args:
        for link in args:
            try:
                parts = link.strip("/").split("/")
                if "discord.com" in link and "channels" in parts:
                    guild_id, channel_id, message_id = parts[-3], parts[-2], parts[-1]
                    channel = await ctx.bot.fetch_channel(int(channel_id))
                    message = await channel.fetch_message(int(message_id))
                    for attachment in message.attachments:
                        if attachment.content_type and attachment.content_type.startswith("image/"):
                            image_urls.append(attachment.url)
            except Exception as e:
                await ctx.send(f"Failed to fetch from link: {link}\nError: {e}")

    if not image_urls:
        await ctx.send("No images found in message or links provided.")
        return

    images = []
    async with aiohttp.ClientSession() as session:
        for url in image_urls:
            async with session.get(url) as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    images.append(img_bytes)
                else:
                    await ctx.send(f"Failed to download image: {url}")

    try:
        await process_images(images, ctx)
    except Exception as e:
        msg = f"""
        An error occurred while processing the images: {e}
        Please make sure the images are clear and contain the full scorecard with minimal obstructions.
        """
        await ctx.send(msg)


bot.run(TOKEN)
