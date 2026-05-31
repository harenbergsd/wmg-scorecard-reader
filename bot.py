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
import shlex

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import utils
import fetch_pars

PARS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pars.csv")
DIFFICULTY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "difficulty.csv")
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
    refresh_course_data.start()
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
        **courses** - List / filter courses by difficulty, name, or exclusion
        **coursewheel** - Spin a wheel of filtered courses        
        **wheel** - Spin a wheel to randomly pick from a list of options
        """,
        inline=False,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=False)
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
    embed.add_field(name="\u200b", value="\u200b", inline=False)
    embed.add_field(
        name="Course list & filtering",
        value="List courses with optional filters:\n"
        "```!larrybot courses easy\n"
        "!larrybot courses hard -WGH -\"meow wolf\" -upside\n"
        "!larrybot courses easy top:10\n"
        "!larrybot courses lair```"
        "**Filters (optional):**\n"
        "• `easy` / `hard` — show only easy or hard variants\n"
        "• `-<code>` or `-<name>` — exclude a course (e.g. `-WGH`, `-\"meow wolf\"`)\n"
        "• `<name>` or `<code>` — fuzzy match by partial name or exact code (e.g. `lair`, `JCH`)\n"
        "• `top:N` / `bottom:N` — Uses WMGT difficulty rankings to limit the result by the N most/least difficult courses\n"
        "No filters: lists all courses",
        inline=False,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=False)
    embed.add_field(
        name="Course wheel",
        value="Spin a wheel of courses matching your filters:\n"
        '```!larrybot coursewheel "Which hard course?" hard -WGH -\"meow wolf\" -upside\n'
        '!larrybot coursewheel "What\'s next?" easy```',
        inline=False,
    )    
    embed.add_field(name="\u200b", value="\u200b", inline=False)
    embed.add_field(
        name="Spin the wheel of fate",
        value="Ask a question and provide options to pick from:\n"
        '```!larrybot wheel "Who goes first?" Alice Bob "Jean Luc"```',
        inline=False,
    )

    await ctx.send(embed=embed)


@bot.command(name="help")
async def top_level_help(ctx):
    await help_command(ctx)


async def _run_wheel(ctx, question, options, selected, result_label):
    """Animate a wheel spin and announce the result."""
    msg = await ctx.send("\U0001f3a1 Spinning the wheel\u2026")
    event_loop = asyncio.get_running_loop()
    gif = await event_loop.run_in_executor(None, _build_wheel_gif, options, selected, question)
    await msg.delete()
    await ctx.send(file=discord.File(gif, filename="wheel.gif"))
    await asyncio.sleep(6)
    await ctx.send(f"\U0001f3a1 The answer to **{question}** is: {result_label}!")


@larrybot.command(name="wheel")
async def wheel_command(ctx, question, option1, *extra):
    options = [option1] + list(extra)
    selected = random.choice(options)
    await _run_wheel(ctx, question, options, selected, f"**{selected}**")


@larrybot.command(name="courses")
async def courses_command(ctx):
    args = _parse_args(ctx)
    courses_data = _load_courses()
    if not courses_data:
        await ctx.send("Course data unavailable (pars.csv not found).")
        return

    difficulty_map = _load_difficulty()
    results = _filter_courses(courses_data, args, difficulty_map)

    if not results:
        await ctx.send("No courses matched your filters.")
        return
    
    col_header = f" {'#':>2}  {'CODE':<4}  {'DIFF':>4}  COURSE\n {'─'*2}  {'─'*4}  {'─'*4}  {'─'*41}"
    rows = []
    for i, (code, name) in enumerate(results, 1):
        diff = difficulty_map.get(code)
        diff_str = f"{diff:.1f}" if diff is not None else " —"
        rows.append(f" {i:>2}  {code:<4}  {diff_str:>4}  {name}")

    # Split across multiple messages only when necessary (Discord 2000-char limit).
    first_header = col_header
    chunks = []
    current = [first_header]
    current_len = 8 + len(first_header) + 1
    for row in rows:
        row_len = len(row) + 1
        if current_len + row_len > 1990:
            chunks.append("```\n" + "\n".join(current) + "\n```")
            current = [row]
            current_len = 8 + row_len
        else:
            current.append(row)
            current_len += row_len
    if current:
        chunks.append("```\n" + "\n".join(current) + "\n```")

    for chunk in chunks:
        await ctx.send(chunk)


@larrybot.command(name="coursewheel")
async def coursewheel_command(ctx):
    parts = _parse_args(ctx)

    if not parts:
        await ctx.send('Usage: `!larrybot coursewheel "<question>" [filters...]` — no filters spins all courses')
        return
    question = parts[0]
    filter_args = parts[1:]
    courses_data = _load_courses()
    if not courses_data:
        await ctx.send("Course data unavailable (pars.csv not found).")
        return

    difficulty_map = _load_difficulty()
    results = _filter_courses(courses_data, filter_args, difficulty_map)

    if not results:
        await ctx.send("No courses matched your filters.")
        return

    if len(results) == 1:
        code, name = results[0]
        await ctx.send(f"Only one course matched: **{name}** (`{code}`)")
        return

    options = [code for code, name in results]
    code_to_name = {code: name for code, name in results}
    selected_code = random.choice(options)
    selected_name = code_to_name[selected_code]
    await _run_wheel(ctx, question, options, selected_code, f"**{selected_name}** (`{selected_code}`)") 


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


def _parse_args(ctx):
    """Split raw message content after the command name using shlex."""
    header = f"{ctx.prefix}{ctx.command.qualified_name}"
    raw = ctx.message.content[len(header):].strip()

    # Normalize fancy Unicode that mobile/Apple keyboards commonly autocorrect to.
    raw = raw.replace("\u201c", '"').replace("\u201d", '"') # Smart/curly quotes
    raw = raw.replace("\u2018", "'").replace("\u2019", "'") # Smart/curly quotes
    raw = raw.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-") # En dash, em dash, mathematical minus
    raw = raw.replace("\u00a0", " ") # Non-breaking space
    try:
        return tuple(shlex.split(raw))
    except ValueError:
        return tuple(raw.split())


def _strip_difficulty(text):
    """Return (has_easy, has_hard, cleaned_text) with easy/hard words removed."""
    has_easy = bool(re.search(r"\beasy\b", text, flags=re.IGNORECASE))
    has_hard = bool(re.search(r"\bhard\b", text, flags=re.IGNORECASE))
    clean = re.sub(r"\b(easy|hard)\b", "", text, flags=re.IGNORECASE)
    # Drop hyphens adjacent to whitespace (dangling separators).
    clean = re.sub(r"\s+-\s*|\s*-\s+", " ", clean).strip()
    return has_easy, has_hard, clean


def _load_courses():
    rows = []
    try:
        with open(PARS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["code"].strip(), row["course"].strip()))
    except FileNotFoundError:
        pass
    return rows


def _load_difficulty():
    """Return {code: stddev} from difficulty.csv. Empty dict if file missing."""
    result = {}
    try:
        with open(DIFFICULTY_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    result[row["code"].strip()] = float(row["stddev"])
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return result


def _filter_courses(courses_data, args, difficulty_map=None):
    """Filter courses_data [(code, name)] by the given args.

    Recognised filter tokens:
      easy / hard       — keep only E- or H-suffix courses
      -<term>           — exclude courses whose code or name contains <term>
      top:<N>           — keep N most difficult courses by difficulty score
      bottom:<N>        — keep N easiest courses by difficulty score
      <anything else>   — fuzzy name / exact code match

    Returns the filtered list.  When top:/bottom: is active the list is
    sorted by difficulty (desc for top, asc for bottom) before slicing.
    """
    difficulty_filter = None  # 'E' or 'H'
    exclusions = []
    name_queries = []
    top_n = None
    bottom_n = None

    for arg in args:
        al = arg.lower().strip()
        if al == "easy":
            difficulty_filter = "E"
        elif al == "hard":
            difficulty_filter = "H"
        elif al.startswith("-") and len(arg) > 1:
            exclusions.append(arg[1:].lower().strip('"'))
        elif al.startswith("top:"):
            try:
                top_n = int(arg[4:])
            except ValueError:
                pass
        elif al.startswith("bottom:"):
            try:
                bottom_n = int(arg[7:])
            except ValueError:
                pass
        else:
            name_queries.append(arg)

    result = list(courses_data)

    if difficulty_filter:
        result = [(code, name) for code, name in result if code.upper().endswith(difficulty_filter)]

    if name_queries:
        matched = []
        for query in name_queries:
            q_upper = query.upper()
            has_easy, has_hard, q_clean = _strip_difficulty(query)
            candidates = [
                (code, name)
                for code, name in result
                if code.upper() == q_upper or utils.word_containment(q_clean, name) >= COURSE_MATCH_MIN_SIMILARITY
            ]
            # If the query named a difficulty, restrict to that variant only.
            # e.g. "Journey easy" only JCE, "Journey hard" only JCH.
            if has_easy:
                candidates = [(code, name) for code, name in candidates if code.upper().endswith("E")]
            if has_hard:
                candidates = [(code, name) for code, name in candidates if code.upper().endswith("H")]
            
            matched += candidates

        result = list(dict.fromkeys(matched))

    for ex in exclusions:
        ex_upper = ex.upper()
        known_codes = {code.upper() for code, _ in courses_data}
        if ex_upper in known_codes:
            # Exact code exclusion — don't accidentally match names that happen
            # to contain the code as a substring (e.g. -OGE shouldn't hit Bogey's)
            result = [(code, name) for code, name in result if code.upper() != ex_upper]
        else:
            # Name exclusion: strip easy/hard but check the code suffix to verify
            has_easy_ex, has_hard_ex, ex_clean = _strip_difficulty(ex)
            result = [
                (code, name)
                for code, name in result
                if utils.word_containment(ex_clean, name) < COURSE_MATCH_MIN_SIMILARITY
                or (has_easy_ex and not code.upper().endswith("E"))
                or (has_hard_ex and not code.upper().endswith("H"))
            ]

    if difficulty_map:
        if top_n is not None:
            result.sort(key=lambda x: difficulty_map.get(x[0], 0.0), reverse=True)
            result = result[:top_n]
        elif bottom_n is not None:
            result.sort(key=lambda x: difficulty_map.get(x[0], float("inf")))
            result = result[:bottom_n]

    return result


def _build_wheel_gif(options, selected, question):
    """Render an animated spin GIF in memory. Returns a BytesIO positioned at 0."""
    n = len(options)
    sweep = 360.0 / n
    selected_idx = options.index(selected)

    # With counterclock=False the centre of wedge i sits at: startangle - (i + 0.5)*sweep
    # We want that to equal 90° (12-o'clock) so the pointer is unambiguous.
    target = 90.0 + (selected_idx + 0.5) * sweep
    # Offset by one wedge so frame 0 is visually distinct from the answer,
    # avoiding the "loop jump" where the last and first frame look identical.
    spin_start = target + 5 * 360.0 + sweep

    colors = [plt.cm.tab20(i % 20) for i in range(n)]
    labels = [o if len(o) <= 14 else o[:13] + "\u2026" for o in options]

    num_frames = 60

    # Quadratic ease-out: smooth deceleration without an overly sharp initial burst.
    def ease_out(t):
        return 1.0 - (1.0 - t) ** 2

    # Uniform frame duration keeps all frames well above the browser/Discord 20 ms
    # minimum GIF delay.  Variable short durations get clamped and make the spin
    # appear to accelerate mid-animation.  60 frames × ~75 ms ≈ 4.5 s total.
    # The last frame holds for 30 s so the GIF freezes on the answer.
    # Last spin frame holds briefly so the wheel visibly comes to rest before the
    # explosion frame; the explosion frame itself is appended after the loop.
    durations = [4.5 / num_frames] * num_frames
    durations[-1] = 1.0

    frames = []
    for i in range(num_frames):
        t = ease_out(i / (num_frames - 1))
        angle = spin_start + t * (target - spin_start)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        wedges, texts = ax.pie(
            [1] * n,
            labels=labels,
            startangle=angle,
            counterclock=False,
            colors=colors,
        )
        ax.text(0, 1.15, "\u25bc", ha="center", va="bottom", fontsize=16, color="red", fontweight="bold")
        ax.axis("equal")
        ax.set_title(question, fontsize=11, pad=22)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    # Explosion frame: winner pops out, no arrow, holds for 10s.
    durations.append(10.0)
    explode = [0.15 if j == selected_idx else 0 for j in range(n)]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    wedges, texts = ax.pie(
        [1] * n,
        labels=labels,
        startangle=target,
        counterclock=False,
        colors=colors,
        explode=explode,
    )
    wedges[selected_idx].set_linewidth(6)
    texts[selected_idx].set_fontweight("bold")
    texts[selected_idx].set_fontsize(13)
    ax.axis("equal")
    ax.set_title(question, fontsize=11, pad=22)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    frames.append(imageio.imread(buf))

    gif = BytesIO()

    pil_frames = [
        Image.fromarray(f).convert("RGB").quantize(colors=256, dither=0)
        for f in frames
    ]
    pil_frames[0].save(
        gif,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=[int(d * 1000) for d in durations],
        loop=0,
        optimize=False,
    )
    gif.seek(0)
    return gif


@tasks.loop(hours=24)
async def refresh_course_data():
    loop = asyncio.get_running_loop()
    try:
        raw_rows = await loop.run_in_executor(None, fetch_pars.fetch_all_rows)
    except Exception as e:
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] fetch failed: {e}; keeping existing pars.csv"
        )
        return
    if not raw_rows:
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] fetch returned no rows; keeping existing pars.csv"
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
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] updated pars.csv with {len(csv_rows)} courses"
        )
    except Exception as e:
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] failed to write pars.csv: {e}")

    # Refresh difficulty.csv independently so a pars failure doesn't skip difficulty.
    try:
        difficulty_rows = await loop.run_in_executor(None, fetch_pars.fetch_difficulty)
        tmp_diff = DIFFICULTY_CSV + ".tmp"
        fetch_pars.write_difficulty_csv(difficulty_rows, tmp_diff)
        os.replace(tmp_diff, DIFFICULTY_CSV)
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] updated difficulty.csv with {len(difficulty_rows)} courses"
        )
    except Exception as e:
        print(
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [refresh_course_data] difficulty fetch failed: {e}; keeping existing difficulty.csv"
        )


if __name__ == "__main__":
    bot.run(TOKEN)
