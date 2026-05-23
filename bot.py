import os
import csv
import datetime
import discord
from dotenv import load_dotenv
from discord.ext import commands, tasks
from scorecard import Scorecard
from PIL import Image
import aiohttp
from io import BytesIO
import hashlib
import asyncio
import threading
import uuid
import time
from collections import defaultdict
import random
import matplotlib
import re

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import utils
import fetch_pars

PARS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pars.csv")
COURSE_MATCH_MIN_SIMILARITY = 0.8


file_lock = threading.Lock()

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
scorecard_cache = {}


@bot.event
async def on_ready():
    refresh_pars.start()
    print(f"Logged in as {bot.user}")


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
        description="Lifeless Algorithm Rapidly Reviewing Your scorecard",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Commands",
        value="""
        **help** - Show this help message        
        **review** - Reviews one or more scorecards
        **course** - Look up a course code or expand it to the full name
        **wheel** - Spin a wheel to randomly pick from a list of options
        """,
        inline=False,
    )
    embed.add_field(
        name="Review scorecards",
        value="Upload image(s) and, in the same message, type:\n"
        "```!larrybot review```\n"
        "Or provide Discord message link(s) with images:\n"
        "```!larrybot review <message-link1> ... <message-linkN>```\n"
        "Or review all scorecards in a channel:\n"
        "```!larrybot review #channel [limit]```\n"
        "`limit` - max number of messages to scan (from most recent, default: all).",
        inline=False,
    )
    embed.add_field(
        name="Course name lookup",
        value="Look up by code or name (fuzzy match supported):\n"
        "```!larrybot course 20E 8BH```"
        '```!larrybot course "Journey - hard"```',
        inline=False,
    )
    embed.add_field(
        name="Spin the wheel of fate",
        value="Ask a question and provide options to pick from:\n"
        '```!larrybot wheel "Who goes first?" Alice Bob Carol```',
        inline=False,
    )
    await ctx.send(embed=embed)


@bot.command(name="help")
async def top_level_help(ctx):
    await help_command(ctx)


@larrybot.command(name="course")
async def course_command(ctx, *args):
    if not args:
        await ctx.send(
            "Usage: `!larrybot course <code or name> [...]`\n"
            'Examples: `!larrybot course 20E 8BH` or `!larrybot course "Journey - hard"`'
        )
        return

    courses_data = _load_courses()
    if not courses_data:
        await ctx.send("Course data unavailable (pars.csv not found).")
        return

    lines = [_lookup_course(arg, courses_data) for arg in args]
    await ctx.send("\n".join(lines))


@larrybot.command(name="wheel")
async def wheel_command(ctx, question, option1, *extra):
    options = [option1] + list(extra)
    selected = random.choice(options)

    msg = await ctx.send("\U0001f3a1 Spinning the wheel\u2026")
    event_loop = asyncio.get_running_loop()
    gif = await event_loop.run_in_executor(None, _build_wheel_gif, options, selected, question)
    await msg.delete()
    await ctx.send(file=discord.File(gif, filename="wheel.gif"))
    await asyncio.sleep(4)
    await ctx.send(f"\U0001f3a1 The answer to **{question}** is: **{selected}**!")


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


async def process_images(images, ctx, msg):
    """images: list of (bytes, source_jump_url) tuples"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, process_images_runner, images, loop, msg)


def process_images_runner(images, loop, msg):
    scorecards = defaultdict(list)
    failures = []  # list of (source_jump_url, error_message)
    msgtext = "Processing images..."
    st = time.time()
    with file_lock:
        standard_contours = utils.load_standard_contours()
        for i, (img, source_url) in enumerate(images):
            t0 = time.time()
            try:
                hashval = hash_bytes(img)
                if hashval in scorecard_cache:
                    s = scorecard_cache[hashval]
                else:
                    s = Scorecard.from_image(img, standard_contours)
                    scorecard_cache[hashval] = s
                scorecards[s.course].append(s.copy())
            except Exception as e:
                failures.append((source_url, str(e)))
                msgtext += f"\nFailed to read card {i+1} of {len(images)}."
                asyncio.run_coroutine_threadsafe(msg.edit(content=msgtext), loop)
                continue

            elapsed = time.time() - t0
            msgtext += f"\nFinished reading card {i+1} of {len(images)} in {elapsed:.2f} seconds."
            asyncio.run_coroutine_threadsafe(msg.edit(content=msgtext), loop)

    if not scorecards:
        failure_text = "No scorecards could be read from the provided images."
        if failures:
            failure_text += f"\n\n**Failed to process {len(failures)} image(s):**\n"
            for source_url, error in failures:
                failure_text += f"\u2022 [Image]({source_url}): {error}\n"
        asyncio.run_coroutine_threadsafe(msg.edit(content=failure_text), loop)
        return

    msgtext += f"\nComputing stats and building tables..."
    asyncio.run_coroutine_threadsafe(msg.edit(content=msgtext), loop)

    imgbufs = []
    csvbufs = []
    courses = []
    for course, scorecard_list in scorecards.items():
        courses.append(course)

        # combine cards
        scorecard = scorecard_list[0]
        for s in scorecard_list[1:]:
            scorecard = scorecard.combine(s)

        sc = scorecard.copy()
        sc.include_pars = True
        sc.include_best = True
        sc = sc.sorted_by_total()
        if sc is not None:
            score_summary = sc.summarize_scores()
            shot_summary = sc.summarize_shots()
            par_comp = sc.compare_to_par().df
            best_comp = sc.compare_to_best().df

            ibuf = BytesIO()
            utils.dfs_to_image(
                [score_summary, shot_summary, sc.df, par_comp, best_comp],
                titles=[
                    "Score Summary",
                    "Shot Summary",
                    "Ordered Scorecard",
                    "Scores Compared to Par",
                    "Scores Compared to Best Hole Score",
                ],
                output_path=ibuf,
            )
            ibuf.seek(0)
            imgbufs.append(ibuf)

            # csv file
            cbuf = BytesIO()
            sc = scorecard.copy()
            sc = sc.sorted_by_total()
            sc.include_pars = True
            sc.df.to_csv(cbuf, index_label=sc.df.index.name)
            cbuf.seek(0)
            csvbufs.append(cbuf)

    filenames = [utils.sanitize_filename(course) for course in courses]
    img_files = [discord.File(b, filename=f"{filenames[i]}.png") for i, b in enumerate(imgbufs)]
    csv_files = [discord.File(b, filename=f"{filenames[i]}.csv") for i, b in enumerate(csvbufs)]

    result_text = ""
    if any("shangri-la" in c.lower() and "hard" in c.lower() for c in courses):
        result_text += (
            "\u26a0\ufe0f **Warning:** Shangri-La - Hard scores may not be accurate due to the "
            "floating lanterns. Please double-check the values.\n\n"
        )
    if failures:
        result_text += f"**Failed to process {len(failures)} image(s):**\n"
        for source_url, error in failures:
            result_text += f"\u2022 [Image]({source_url}): {error}\n"

    asyncio.run_coroutine_threadsafe(msg.edit(content=result_text, attachments=img_files + csv_files), loop)

    print(f"Processed {len(images)} images ({len(failures)} failed) in {time.time() - st:.2f} seconds.")


def get_image_attachments(message):
    """Return list of (download_url, jump_url) for image attachments on a message."""
    return [
        (a.url, message.jump_url) for a in message.attachments if a.content_type and a.content_type.startswith("image/")
    ]


async def parse_review_args(ctx, args):
    """Parse args into (channel_to_scan, scan_limit, image_sources) or raise."""
    image_sources = []
    channel_to_scan = None
    scan_limit = None

    for arg in args:
        if arg.startswith("<#") and arg.endswith(">"):
            channel_id = int(arg[2:-1])
            channel_to_scan = await ctx.bot.fetch_channel(channel_id)
        elif "discord.com" in arg and "channels" in arg:
            parts = arg.strip("/").split("/")
            channel_id, message_id = parts[-2], parts[-1]
            channel = await ctx.bot.fetch_channel(int(channel_id))
            message = await channel.fetch_message(int(message_id))
            image_sources.extend(get_image_attachments(message))
        else:
            scan_limit = int(arg)
            if scan_limit <= 0:
                raise ValueError("Limit must be a positive number.")

    return channel_to_scan, scan_limit, image_sources


async def scan_channel_images(channel, limit=None):
    """Scan a channel's history and return (image_sources, message_count)."""
    image_sources = []
    message_count = 0
    oldest_first = limit is None
    async for message in channel.history(limit=limit, oldest_first=oldest_first):
        message_count += 1
        image_sources.extend(get_image_attachments(message))
    if not oldest_first:
        image_sources.reverse()
    return image_sources, message_count


async def download_images(image_sources):
    """Download images from URLs. Returns list of (bytes, jump_url) tuples."""
    images = []
    async with aiohttp.ClientSession() as session:
        for url, source in image_sources:
            async with session.get(url) as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    images.append((img_bytes, source))
    return images


@larrybot.command(name="review")
async def review_command(ctx, *args):
    # Collect images attached to the command message
    image_sources = get_image_attachments(ctx.message)

    # Parse remaining args for channel mentions, message links, scan limit
    try:
        channel_to_scan, scan_limit, link_sources = await parse_review_args(ctx, args)
        image_sources.extend(link_sources)
    except (ValueError, IndexError) as e:
        await ctx.send(f"Bad argument: {e}")
        return
    except Exception as e:
        await ctx.send(f"Error parsing arguments: {e}")
        return

    # Scan channel if one was explicitly specified
    if channel_to_scan is not None:
        limit_desc = "all messages" if scan_limit is None else f"up to {scan_limit} messages"
        msg = await ctx.send(f"Scanning {limit_desc} in {channel_to_scan.mention} for images...")
        try:
            channel_sources, message_count = await scan_channel_images(channel_to_scan, scan_limit)
        except discord.Forbidden:
            await msg.edit(content=f"I don't have permission to read message history in {channel_to_scan.mention}.")
            return
        except Exception as e:
            await msg.edit(content=f"Error scanning channel: {e}")
            return

        image_sources.extend(channel_sources)
        if not image_sources:
            await msg.edit(content=f"No images found in {channel_to_scan.mention} (scanned {message_count} messages).")
            return

        await msg.edit(content=f"Found {len(image_sources)} images in {message_count} messages. Downloading...")
    elif not image_sources:
        await ctx.send("No images found. Attach images, provide message links, or specify a #channel to scan.")
        return
    else:
        msg = await ctx.send("Downloading images...")

    images = await download_images(image_sources)
    if not images:
        await msg.edit(content="Failed to download any images.")
        return

    await msg.edit(content=f"Processing {len(images)} images...")
    try:
        await process_images(images, ctx, msg)
    except Exception as e:
        msgtext = (
            f"An error occurred while processing the images: {e}\n"
            "Please make sure the images are clear and contain the full scorecard with minimal obstructions."
        )
        await msg.edit(content=msgtext)


def _load_courses():
    rows = []
    try:
        with open(PARS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["code"].strip(), row["course"].strip()))
    except FileNotFoundError:
        pass
    return rows


def _lookup_course(arg, courses_data):
    arg = arg.strip()

    # Exact code match (case-insensitive)
    for code, name in courses_data:
        if code.upper() == arg.upper():
            return f"`{code}` → {name}"

    # Exact name match (case-insensitive)
    for code, name in courses_data:
        if name.lower() == arg.lower():
            return f"{name} → `{code}`"

    # Fuzzy name match — word containment: fraction of query words found in course name
    arg = re.sub(r"\b(easy)\b", "", arg, flags=re.IGNORECASE).strip()  # disregard 'easy'
    best_sim, best = 0.0, None
    for code, name in courses_data:
        sim = utils.word_containment(arg, name)
        if sim > best_sim:
            best_sim, best = sim, (code, name)

    if best and best_sim >= COURSE_MATCH_MIN_SIMILARITY:
        code, name = best
        return f"{name} → `{code}`"

    return f"❓ No match found for `{arg}`"


def _build_wheel_gif(options, selected, question):
    """Render an animated spin GIF in memory. Returns a BytesIO positioned at 0."""
    n = len(options)
    sweep = 360.0 / n
    selected_idx = options.index(selected)

    # With counterclock=False the centre of wedge i sits at: startangle - (i + 0.5)*sweep
    # We want that to equal 90° (12-o'clock) so the pointer is unambiguous.
    target = 90.0 + (selected_idx + 0.5) * sweep
    # Begin several full rotations ahead so the spin is clearly visible.
    spin_start = target + 5 * 360.0

    colors = [plt.cm.tab20(i % 20) for i in range(n)]
    labels = [o if len(o) <= 14 else o[:13] + "\u2026" for o in options]

    num_frames = 60

    # Ease-out quadratic: decelerates more gradually than cubic, giving a drawn-out coast to rest.
    def ease_out(t):
        return 1.0 - (1.0 - t) ** 2

    # Build per-frame durations: shorter early (fast spin), longer near end (slow stop).
    # With quadratic ease the velocity ~ (1-t), so duration ~ 1/(1-t).
    # We clamp to avoid division by zero on the very last frame.
    raw_durations = [1.0 / max(1.0 - i / (num_frames - 1), 0.03) for i in range(num_frames)]
    total = sum(raw_durations)
    # Scale so the whole animation lasts ~4 s.
    durations = [d / total * 4.0 for d in raw_durations]

    frames = []
    for i in range(num_frames):
        t = ease_out(i / (num_frames - 1))
        angle = spin_start + t * (target - spin_start)
        is_last = i == num_frames - 1

        fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
        wedges, texts = ax.pie(
            [1] * n,
            labels=labels,
            startangle=angle,
            counterclock=False,
            colors=colors,
        )
        if is_last:
            wedges[selected_idx].set_edgecolor("gold")
            wedges[selected_idx].set_linewidth(5)
            texts[selected_idx].set_fontweight("bold")

        # Pointer: a downward triangle just above the 12-o'clock position.
        # Pie radius is 1.0 in data coords; place pointer at y=1.15.
        ax.text(0, 1.15, "\u25bc", ha="center", va="bottom", fontsize=16, color="red", fontweight="bold")

        ax.axis("equal")
        ax.set_title(question, fontsize=11, pad=22)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    gif = BytesIO()
    imageio.mimsave(gif, frames, format="gif", duration=durations)
    gif.seek(0)
    return gif


@tasks.loop(hours=24)
async def refresh_pars():
    loop = asyncio.get_running_loop()
    try:
        raw_rows = await loop.run_in_executor(None, fetch_pars.fetch_all_rows)
    except Exception as e:
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_pars] fetch failed: {e}; keeping existing pars.csv"
        )
        return
    if not raw_rows:
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_pars] fetch returned no rows; keeping existing pars.csv"
        )
        return
    try:
        csv_rows = fetch_pars.build_csv_rows(raw_rows)
        # Write to a temp file first, then atomically replace the live CSV so
        # concurrent readers never see a truncated or partial file.
        tmp = PARS_CSV + ".tmp"
        fetch_pars.write_csv(csv_rows, tmp)
        os.replace(tmp, PARS_CSV)
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_pars] updated pars.csv with {len(csv_rows)} courses"
        )
    except Exception as e:
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_pars] failed to write pars.csv: {e}")


bot.run(TOKEN)
