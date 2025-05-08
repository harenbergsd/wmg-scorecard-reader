import os
import discord
from dotenv import load_dotenv
from discord.ext import commands
from scorecard import Scorecard
from PIL import Image
import aiohttp
from io import BytesIO
import hashlib
import asyncio
import threading
import uuid
import time

import misc

file_lock = threading.Lock()

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix="!", intents=intents)
scorecard_cache = {}


class ScorecardError(Exception):
    pass


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


@bot.group(invoke_without_command=True)
async def larrybot(ctx):
    await ctx.send("Usage: `!larrybot <command> [option(s)] [message_link(s)]`\nType `!larrybot help` for details.")


@larrybot.command(name="help")
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
        "```!larrybot review```\n"
        "Or provide a (or multiple) Discord message link(s) with images:\n"
        "```!larrybot review <message-link1> ... <message-linkN>```",
        inline=False,
    )
    embed.set_footer(text="LARRY: Lifeless Algorithm Rapidly Reviewing Your scorecard")
    await ctx.send(embed=embed)


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
    await loop.run_in_executor(None, _process_images_runner, images, ctx, loop)


def _process_images_runner(images, ctx, loop):
    tmp_filename = None
    try:
        tmp_filename = f"{uuid.uuid4().hex}.png"
        _process_images_sync(images, tmp_filename, ctx, loop)
    except Exception as e:
        raise
    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def _process_images_sync(images, tmp_filename, ctx, loop):
    st = time.time()
    with file_lock:
        standard_contours = misc.load_standard_contours()
        scorecard = None
        try:
            for i, img in enumerate(images):
                hashval = hash_bytes(img)
                if hashval in scorecard_cache:
                    s = scorecard_cache[hashval]
                else:
                    s = Scorecard.from_image(img, standard_contours)
                    scorecard_cache[hashval] = s
                scorecard = s.copy() if scorecard is None else scorecard.combine(s)
        except Exception as e:
            raise ScorecardError(f"Failed to process image {i+1} of {len(images)}: {e}")

    sc = scorecard.copy()
    sc.include_pars = True
    sc.include_best = True
    sc = sc.sorted_by_total()
    if sc is not None:
        text = sc.course
        df = sc.summarize_scores()
        max_rows = 10  # Limit to 10 rows for Discord message size
        if len(df) > max_rows:
            text += f"\n(Summary tables limited to 10 rows due to Discord message size limits)"
        text += f"\n```{misc.df_to_str(df, max_rows=max_rows)}```"

        df = sc.summarize_shots()
        text += f"```{misc.df_to_str(df, max_rows=max_rows)}```"

        par_comp = sc.compare_to_par().df
        best_comp = sc.compare_to_best().df
        titles = ["Ordered Scorecard", "Scores Compared to Par", "Scores Compared to Best Hole Score"]
        misc.dfs_to_image([sc.df, par_comp, best_comp], titles=titles, output_path=tmp_filename)

        # csv file
        b = BytesIO()
        sc = scorecard.copy()
        sc = sc.sorted_by_total()
        sc.include_pars = True
        sc.df.to_csv(b, index_label=sc.df.index.name)
        b.seek(0)
        csv_file = discord.File(b, filename="scorecard.csv")

        future = asyncio.run_coroutine_threadsafe(_send_result(ctx, text, tmp_filename, csv_file), loop)
        future.result()

    print(f"Processed {len(images)} images in {time.time() - st:.2f} seconds.")


async def _send_result(ctx, text, tmp_filename, csv_file=None):
    if len(text) > 1950:
        text = ""
    with open(tmp_filename, "rb") as f:
        files = [discord.File(f)]
        if csv_file:
            files.append(csv_file)
        await ctx.send(text, files=files)


@larrybot.command(name="review")
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
