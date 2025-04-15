import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from playwright.async_api import (
    Page,
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError, expect,
)
from pydantic import BaseModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG_FILE = "interface_config.json"
DEFAULT_STREAM_CHECK_INTERVAL = 0.1 # Seconds to wait between checking for stream updates
DEFAULT_COMPLETION_TIMEOUT = 120.0  # Max seconds to wait for non-streaming completion
DEFAULT_STREAM_IDLE_TIMEOUT = 5.0   # Seconds of no new content before declaring stream finished (if no completion_indicator)

# --- Global State (managed by lifespan) ---
app_state: Dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "page": None,
    "config": None,
    "current_task": None, # Store the current generation task for cancellation
    "lock": asyncio.Lock() # Prevent concurrent generations if the website UI can't handle it
}

# --- Pydantic Models ---
class CompletionRequest(BaseModel):
    messages: list
    stream: bool = False
    temperature: Optional[float] = None
    output_length: Optional[int] = None # Renamed for clarity, adjust as needed
    top_p: Optional[float] = None
    # Add any other parameters the website UI might support
    # e.g., max_tokens: Optional[int] = None
    # model: Optional[str] = None # If there's a model selector UI element

class CompletionResponseChunk(BaseModel):
    id: str = "chatcmpl-xxx" # Mimic OpenAI format
    object: str = "chat.completion.chunk"
    created: int = 0 # Timestamp, can be added
    model: str = "website-model" # Placeholder
    choices: list[dict]

class CompletionResponse(BaseModel):
    id: str = "chatcmpl-yyy" # Mimic OpenAI format
    object: str = "chat.completion"
    created: int = 0 # Timestamp, can be added
    model: str = "website-model" # Placeholder
    choices: list[dict]
    usage: Optional[dict] = None # Can't easily get token usage

# --- Helper Functions ---

def load_config() -> Dict[str, str]:
    """Loads XPath configuration from the JSON file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {CONFIG_FILE}")
        # Basic validation
        required_keys = ["input_settings", "send_button", "stop_button", "response_area"]
        if not all(key in config for key in required_keys):
             raise ValueError(f"Config file missing one or more required keys: {required_keys}")
        return config
    except FileNotFoundError:
        logging.error(f"ERROR: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"ERROR: Configuration file '{CONFIG_FILE}' contains invalid JSON.")
        sys.exit(1)
    except ValueError as e:
         logging.error(f"ERROR: Configuration error: {e}")
         sys.exit(1)

async def safe_locate(page: Page, xpath: Optional[str], timeout: int = 5000):
    """Safely locates an element, returning None if not found or xpath is None."""
    if not xpath:
        return None
    try:
        return page.locator(xpath).first # Use .first to avoid ambiguity if XPath matches multiple
    except PlaywrightError as e:
        logging.warning(f"Could not locate element with XPath: {xpath}. Error: {e}")
        return None

async def set_ui_parameters(page: Page, config:dict, request: CompletionRequest):
    """Attempts to set UI parameters based on request and config. Very basic example."""
    # This needs significant expansion based on the actual website UI
    # Example for a simple input field:
    if request.temperature is not None and config.get("temperature_setting"):
        temp_input = await safe_locate(page, config["temperature_setting"]['position'])

        # handle temperature inputs
        if temp_input:
            try:
                await temp_input.fill(str(request.temperature))

                # Alternate: directly set the value.
                #await temp_input.evaluate(f'(element, value) => {{ element.{config['temperature_setting']['name']} = value; }}', str(request.temperature))
                logging.info(f"Set temperature UI to: {request.temperature}")
            except PlaywrightError as e:
                logging.warning(f"Failed to set temperature UI: {e}")
        else:
            print("did not find temperature input position!")

    if request.top_p is not None and config.get("top_p_setting"):
        top_p_input_xpath = config["top_p_setting"]['position']
        top_p_input = await safe_locate(page, top_p_input_xpath)

        # handle top P inputs
        if top_p_input:
            try:
                await top_p_input.fill(str(request.top_p))
                logging.info(f"Set top_p UI to: {request.top_p}")
            except PlaywrightError as e:
                logging.warning(f"Failed to set top_p UI: {e}")


    if request.output_length is not None and config.get("output_length_input"):
        length_input_xpath = config.get("output_length_input")['position']
        length_input = await safe_locate(page, length_input_xpath)
        if length_input:
            try:
                await length_input.fill(str(request.output_length))
                logging.info(f"Set output_length UI to: {request.output_length}")
            except PlaywrightError as e:
                logging.warning(f"Failed to set output_length UI: {e}")

    # Add logic for sliders (e.g., using bounding_box and mouse.move/down/up)
    # Add logic for dropdowns (e.g., click to open, click option)

async def stop_generation(page: Page, config: Dict[str, str]):
    """Attempts to click the stop generation button."""
    stop_button_xpath = config.get("stop_button")
    logging.info("Attempting to stop generation...")
    stop_button = await safe_locate(page, stop_button_xpath)
    if stop_button:
        try:
            await stop_button.click(timeout=2000)
            logging.info("Clicked stop generation button.")
            return True
        except PlaywrightError as e:
            logging.warning(f"Could not click stop button (maybe already stopped?): {e}")
            # Fallthrough, maybe it stopped on its own
    else:
        logging.warning("Stop button XPath not configured or element not found.")
    return False


async def stream_response_generator(page: Page, config: Dict[str, str]) -> AsyncGenerator[str, None]:
    """Generator that yields new text chunks from the response area."""
    response_area_xpath = config["response_area"]
    chunk_indicator_xpath = config.get("response_chunk_indicator") # Optional
    completion_indicator_xpath = config.get("completion_indicator") # Optional

    last_text = ""
    last_change_time = asyncio.get_event_loop().time()
    stream_check_interval = DEFAULT_STREAM_CHECK_INTERVAL
    idle_timeout = DEFAULT_STREAM_IDLE_TIMEOUT

    response_locator = await safe_locate(page, response_area_xpath)
    if not response_locator:
        yield f"data: {json.dumps({'error': 'Response area not found'})}\n\n"
        return

    while True:
        await asyncio.sleep(stream_check_interval)

        try:
            # --- Check for completion ---
            if completion_indicator_xpath:
                completion_element = await safe_locate(page, completion_indicator_xpath, timeout=100) # Short timeout
                # Check if element exists or is visible/enabled (adapt condition as needed)
                if completion_element and await completion_element.is_visible():
                    logging.info("Completion indicator found. Ending stream.")
                    break # Exit loop, generation is complete

            # --- Check for new content ---
            current_text = ""
            if chunk_indicator_xpath:
                 # More efficient: Check only the element likely to contain the newest chunk
                 chunk_locator = await safe_locate(page, chunk_indicator_xpath, timeout=100)
                 if chunk_locator:
                     current_text = await chunk_locator.text_content() or ""
                     # This logic assumes the *entire* last chunk is in the indicator
                     # You might need to combine this with the full response area text
                     # For simplicity now, we'll rely on the full text check below
                     # but log if we find the chunk indicator
                     # logging.debug(f"Chunk indicator text: {current_text}")
                 # Fall back to full text if chunk indicator fails
                 current_text = await response_locator.text_content() or ""

            else:
                # Less efficient: Get the entire response text
                 current_text = await response_locator.text_content() or ""


            if current_text != last_text:
                new_text = current_text[len(last_text):]
                if new_text:
                    logging.debug(f"Streaming new chunk: {new_text[:50]}...")
                    # Send SSE formatted data
                    chunk_data = CompletionResponseChunk(
                        choices=[{"delta": {"content": new_text}, "index": 0, "finish_reason": None}]
                    ).model_dump_json() # Use model_dump_json for Pydantic v2+
                    yield f"data: {chunk_data}\n\n"
                    last_text = current_text
                    last_change_time = asyncio.get_event_loop().time()

            # --- Check for idle timeout (fallback if no completion indicator) ---
            elif not completion_indicator_xpath:
                 if (asyncio.get_event_loop().time() - last_change_time) > idle_timeout:
                    logging.info(f"Stream idle for {idle_timeout}s. Assuming completion.")
                    break

        except PlaywrightTimeoutError:
            logging.warning("Timeout while checking for response update. Continuing...")
        except PlaywrightError as e:
            logging.error(f"Playwright error during streaming: {e}")
            yield f"data: {json.dumps({'error': f'Playwright error: {e}'})}\n\n"
            break
        except Exception as e:
            logging.exception(f"Unexpected error during streaming: {e}")
            yield f"data: {json.dumps({'error': f'Server error: {e}'})}\n\n"
            break

    # Send the final DONE signal for SSE
    yield "data: [DONE]\n\n"
    logging.info("Streaming finished.")


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic."""
    global app_state
    logging.info("Application startup...")

    # 1. Load Config
    app_state["config"] = load_config()

    # 2. Connect to Browser
    app_state["playwright"] = await async_playwright().start()
    p = app_state["playwright"]

    connected = False
    while not connected:
        # --- Realistic Browser Detection ---
        # Playwright cannot easily *list* all running browsers with debug ports.
        # The user MUST launch the browser with the debugging port enabled FIRST.
        # Example: google-chrome --remote-debugging-port=9222
        #          msedge --remote-debugging-port=9223
        # We prompt the user for the CDP endpoint URL.
        # Common default is http://localhost:9222 (or 9223 etc.)

        if not app_state['config']['last_used_debug_link']:

            cdp_url = input(
                "Please ensure your browser was started with --remote-debugging-port=<port>.\n"
                "Enter the browser's remote debugging URL (e.g., http://localhost:9222): "
            )
            cdp_url = cdp_url.strip()
        else:
            cdp_url = app_state['config']['last_used_debug_link']


        if not cdp_url.startswith(("http://", "https://")):
             cdp_url = "http://" + cdp_url # Assume http if not specified

        try:
            logging.info(f"Attempting to connect to browser at {cdp_url}...")
            # Note: Playwright needs the CDP endpoint, often served over HTTP,
            # it internally uses the WebSocketDebuggerUrl found there.
            app_state["browser"] = await p.chromium.connect_over_cdp(cdp_url, timeout=15000)
            logging.info(f"Successfully connected to browser: {app_state['browser'].version}")

            # Use the first available context/page, or prompt user if multiple exist
            contexts = app_state["browser"].contexts
            if not contexts:
                logging.warning("No existing browser contexts found. Using default context.")
                # Sometimes connect_over_cdp doesn't immediately show contexts if none were interacted with *after* launch
                # We might need to create one or attach to the first page directly if possible.
                # Let's try getting the first page directly as a fallback.
                all_pages = []
                for ctx in contexts:
                    all_pages.extend(ctx.pages)

                if not all_pages and len(contexts)>0: # If context exists but no pages listed, try default context pages
                     all_pages = contexts[0].pages

                if not all_pages:
                     # As a last resort, maybe create a new page in the default context? Risky.
                     logging.warning("No pages found. Attempting to use the first context's presumed default page.")
                     # This might fail if the browser wasn't launched with an initial tab.
                     if contexts:
                        app_state["page"] = contexts[0].pages[0] if contexts[0].pages else await contexts[0].new_page()
                     else:
                         # If truly no context, we might need to create one? Unlikely scenario for connect_over_cdp
                         logging.error("Could not find or create a page to attach to.")
                         raise PlaywrightError("No suitable page found in connected browser.")

                elif len(all_pages) == 1:
                    app_state["page"] = all_pages[0]
                    logging.info(f"Attached to the only available page: {app_state['page'].url}")
                else:
                    print("\nMultiple pages found. Please select the target page:")
                    for i, page in enumerate(all_pages):
                        title = await page.title()
                        print(f"  {i}: {page.url} - '{title}'")
                    while True:
                        try:
                            choice = int(input(f"Enter page number (0-{len(all_pages)-1}): "))
                            if 0 <= choice < len(all_pages):
                                app_state["page"] = all_pages[choice]
                                logging.info(f"Attached to selected page: {app_state['page'].url}")
                                break
                            else:
                                print("Invalid choice.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
            elif len(contexts) == 1 and len(contexts[0].pages) == 1:
                 app_state["page"] = contexts[0].pages[0]
                 logging.info(f"Attached to the only available page: {app_state['page'].url}")
            else: # Multiple contexts or pages within contexts
                 all_pages = []
                 page_to_context = {}
                 print("\nMultiple browser contexts/pages found. Please select the target page:")
                 current_index = 0
                 for ctx_idx, context in enumerate(contexts):
                      print(f" Context {ctx_idx}:")
                      if not context.pages:
                           print("    (No pages in this context)")
                           continue
                      for page_idx, page in enumerate(context.pages):
                           title = await page.title()
                           print(f"  {current_index}: {page.url} - '{title}'")
                           all_pages.append(page)
                           page_to_context[current_index] = context
                           current_index += 1

                 if not all_pages:
                     logging.error("No pages found across all contexts.")
                     raise PlaywrightError("No suitable page found in connected browser.")

                 while True:
                      try:
                          choice = int(input(f"Enter page number (0-{len(all_pages)-1}): "))
                          if 0 <= choice < len(all_pages):
                              app_state["page"] = all_pages[choice]
                              logging.info(f"Attached to selected page: {app_state['page'].url}")
                              break
                          else:
                              print("Invalid choice.")
                      except ValueError:
                          print("Invalid input. Please enter a number.")

            # Bring the selected page to the front (optional, but helpful)
            await app_state["page"].bring_to_front()
            connected = True

        except PlaywrightTimeoutError:
            logging.error(f"Timeout connecting to {cdp_url}. Is the browser running with the correct port? Is the URL correct?")
        except PlaywrightError as e:
             logging.error(f"Playwright Error connecting to browser: {e}")
             logging.error("Ensure the browser is running and the debugging port is open.")
             if "Browser closed" in str(e):
                 sys.exit("Browser seems to have closed unexpectedly.") # Exit if connection implies browser is gone
        except Exception as e:
            logging.exception(f"Unexpected error during browser connection: {e}")
            print("An unexpected error occurred. Please check logs.")
            # Decide whether to retry or exit
            retry = input("Retry connection? (y/n): ").lower()
            if retry != 'y':
                sys.exit(1)

    # Yield control to the FastAPI application
    logging.info("Browser connected. Starting API server...")
    yield
    # --- Shutdown ---
    logging.info("Application shutdown...")
    if app_state["current_task"]:
        logging.info("Cancelling any ongoing generation task...")
        app_state["current_task"].cancel()
        # Optionally try to click stop button again during shutdown
        if app_state["page"] and app_state["config"]:
            try:
                await stop_generation(app_state["page"], app_state["config"])
            except Exception as e:
                logging.warning(f"Error trying to stop generation during shutdown: {e}")

    if app_state["browser"]:
        logging.info("Disconnecting from browser...")
        try:
            # Don't close the browser, just disconnect
            await app_state["browser"].disconnect()
        except PlaywrightError as e:
            logging.warning(f"Error disconnecting from browser (might have already closed): {e}")
    if app_state["playwright"]:
        logging.info("Stopping Playwright...")
        await app_state["playwright"].stop()
    logging.info("Shutdown complete.")


# --- FastAPI App and Endpoint ---
app = FastAPI(lifespan=lifespan, title="Web UI API Bridge")

@app.get("/v1/models")
async def get_models():
    return "Please set model on the popup page."

@app.post("/v1/chat/completions", response_model=None) # Response model handled dynamically
async def create_completion(request: CompletionRequest, http_request: Request):
    """
    Handles OpenAI-style requests and interacts with the browser UI.
    Supports streaming and cancellation.
    """
    logging.info('got request')
    logging.info(request.model_dump_json())


    page = app_state.get("page")
    config = app_state.get("config")
    lock = app_state.get("lock")

    if not page or not config or not lock:
        raise HTTPException(status_code=503, detail="Service not ready. Browser/Config not initialized.")

    if page.is_closed():
         raise HTTPException(status_code=503, detail="Browser page is closed. Please restart the application and browser.")

    async with lock: # Ensure only one generation happens at a time
        # --- Check for existing task (optional, might be handled by lock) ---
        if app_state.get("current_task") and not app_state["current_task"].done():
             raise HTTPException(status_code=429, detail="Another generation is already in progress.")

        # --- Prepare UI ---
        prompt_area = await safe_locate(page, config["prompt_textarea"])
        send_button = await safe_locate(page, config["send_button"])

        # here is the refresh button
        refresh_btn = await safe_locate(page, config['clear_history'])

        if not prompt_area or not send_button:
            raise HTTPException(status_code=500, detail="Could not find essential UI elements (prompt area or send button). Check config and website state.")

        try:

            # FIRST -> click the refresh
            logging.info("awaiting refresh button click")
            await refresh_btn.click()



            # --- Set Parameters (if applicable) ---
            await set_ui_parameters(page, config, request) # Best effort

            # --- Enter Prompt ---
            logging.info("Filling prompt...")

            # concat all msgs in request.
            final_str = ''
            for message in request.messages:
                final_str += f'''\n\n{message['role']} ：{message['content']} '''

            await prompt_area.fill(final_str)
            await asyncio.sleep(0.1) # Small delay in case of JS listeners

            # --- Click Send ---
            logging.info("Clicking send button...")
            # Clear previous response area content *before* clicking send? Sometimes needed.
            # response_locator = await safe_locate(page, config["response_area"])
            # if response_locator: await response_locator.evaluate('node => node.textContent = ""')

            await send_button.click()
            logging.info("Generation request sent to UI.")

            # Then, set the in_progress flag to be True.
            # In progress flag:
            # If it is present & button text = "stop" then it is in progress.
            in_progress = [True]

            # Then, finished = if the button text = True


            # --- Handle Response (Streaming or Non-Streaming) ---
            generation_task = None
            try:
                if request.stream:
                    logging.info("Streaming response...")
                    generator = stream_response_generator(page, config)
                    # Store the task so it can be cancelled
                    async def stream_wrapper():
                         try:
                             async for chunk in generator:
                                 # Check for client disconnect within the wrapper
                                 if await http_request.is_disconnected():
                                     logging.info("Client disconnected during stream.")
                                     raise asyncio.CancelledError()
                                 yield chunk
                         except asyncio.CancelledError:
                              logging.info("Streaming task cancelled.")
                              # Attempt to stop generation in the UI
                              await stop_generation(page, config)
                              raise # Re-raise to notify StreamingResponse
                         finally:
                             app_state["current_task"] = None # Clear task on completion/cancellation

                    app_state["current_task"] = asyncio.create_task(stream_wrapper().__anext__()) # Start the task slightly
                    # Use text/event-stream for Server-Sent Events
                    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

                else: # Non-streaming
                    logging.info("Waiting for non-streaming response...")
                    response_area_xpath = config["response_area"]
                    completion_indicator_xpath = config.get("completion_indicator")
                    response_locator = await safe_locate(page, response_area_xpath)
                    if not response_locator:
                         raise HTTPException(status_code=500, detail="Response area element not found.")

                    final_text = ""
                    # Store the task so it can be cancelled

                    # Now, handle the completion await.

                    async def wait_for_completion():
                        nonlocal final_text
                        start_time = asyncio.get_event_loop().time()
                        try:
                            if completion_indicator_xpath:
                                comp_indicator = await safe_locate(page, completion_indicator_xpath)
                                if comp_indicator:

                                    running_regex = re.compile(r"stop", re.IGNORECASE)
                                    stop_regex = re.compile(r'run', re.IGNORECASE)


                                    logging.info(f"Waiting for completion indicator: {completion_indicator_xpath}")
                                    await asyncio.sleep(0.1)

                                    # if it is already running, we will wait for it to become run again.
                                    # otherwise it will become stop.

                                    # brief test: let it sleep for 30
                                    # Then print the results

                                    btn_text = await comp_indicator.text_content()
                                    print(btn_text)


                                    await expect(comp_indicator).to_contain_text('Run', ignore_case=True, timeout = 100000000)

                                    logging.info("Completion indicator appeared.")

                                else:
                                     logging.warning("Completion indicator XPath defined but element not found initially. Falling back to timeout/stop button check.")
                                     # Fallback: Wait for stop button to disappear OR timeout
                                     stop_button = await safe_locate(page, config["stop_button"])
                                     if stop_button:
                                          await stop_button.wait_for(state="hidden", timeout=DEFAULT_COMPLETION_TIMEOUT * 1000)
                                          logging.info("Stop button disappeared, assuming completion.")
                                     else: # No stop button, just wait
                                          await page.wait_for_timeout(DEFAULT_COMPLETION_TIMEOUT * 1000) # Less reliable
                            else:
                                 # No completion indicator: Best effort - wait for stop button to disappear or timeout
                                stop_button_xpath = config.get("stop_button")
                                if stop_button_xpath:
                                    stop_button = await safe_locate(page, stop_button_xpath)
                                    if stop_button:
                                        try:
                                            logging.info("Waiting for stop button to disappear...")
                                            await stop_button.wait_for(state="hidden", timeout=DEFAULT_COMPLETION_TIMEOUT * 1000)
                                            logging.info("Stop button disappeared, assuming completion.")
                                        except PlaywrightTimeoutError:
                                            logging.warning("Timeout waiting for stop button to disappear. Stopping generation attempt.")
                                            await stop_generation(page, config) # Try to stop it if it timed out
                                    else: # Stop button configured but not found now? Maybe it finished fast. Wait a bit.
                                         logging.info("Stop button not found after sending, assuming completion or using timeout.")
                                         await page.wait_for_timeout(3000) # Short wait
                                else:
                                    # Absolute fallback: Just wait a fixed time (least reliable)
                                    logging.warning("No completion indicator or stop button configured. Waiting for fixed timeout.")
                                    await page.wait_for_timeout(DEFAULT_COMPLETION_TIMEOUT * 1000)


                            # Get final text *after* waiting
                            final_text = await response_locator.text_content() or ""
                            logging.info(f"Non-streaming generation completed. Length: {len(final_text)}")

                        except asyncio.CancelledError:
                            logging.info("Non-streaming task cancelled by client.")
                            await stop_generation(page, config)
                            raise # Re-raise
                        except PlaywrightTimeoutError:
                             logging.error(f"Timeout waiting for non-streaming completion after {DEFAULT_COMPLETION_TIMEOUT}s.")
                             await stop_generation(page, config) # Attempt to stop runaway generation
                             raise HTTPException(status_code=504, detail="Generation timed out.")
                        except PlaywrightError as e:
                            logging.error(f"Playwright error waiting for completion: {e}")
                            raise HTTPException(status_code=500, detail=f"Browser interaction error: {e}")
                        finally:
                            app_state["current_task"] = None # Clear task

                    generation_task = asyncio.create_task(wait_for_completion())
                    app_state["current_task"] = generation_task

                    # Wait for the task, handling potential disconnects
                    while not generation_task.done():
                        if await http_request.is_disconnected():
                            logging.info("Client disconnected during non-streaming wait.")
                            generation_task.cancel()
                            # Wait briefly for cancellation handler to run
                            await asyncio.sleep(0.5)
                            # No response can be sent now, FastAPI handles the broken pipe.
                            # The finally block in wait_for_completion will try to stop the UI generation.
                            # We need to raise here to stop processing in the endpoint.
                            # However, FastAPI might have already closed the connection.
                            # Returning None or letting it fall through might be okay.
                            # Let's raise a standard exception FastAPI might ignore gracefully on disconnect.
                            raise asyncio.CancelledError("Client disconnected")
                        await asyncio.sleep(0.1)

                    # If we get here, the task finished normally or errored (handled internally)
                    await generation_task # Raise exceptions if wait_for_completion failed internally

                    # Return final response
                    response_data = CompletionResponse(
                        choices=[{"message": {"role": "assistant", "content": final_text}, "index": 0, "finish_reason": "stop"}]
                    )
                    return response_data.model_dump() # Use model_dump for Pydantic v2+

            except asyncio.CancelledError:
                 # This catches cancellation triggered by disconnect *before* returning StreamingResponse
                 # or during the non-streaming wait loop.
                 logging.info("Request cancelled by client disconnect.")
                 # Stop generation if not already handled
                 if app_state.get("page") and app_state.get("config") and not await stop_generation(app_state["page"], app_state["config"]):
                      logging.warning("Could not stop generation during cancellation cleanup.")
                 # Can't send a response here as the client is gone.
                 # Re-raising might cause internal FastAPI errors, maybe just return?
                 # Let FastAPI handle the disconnect gracefully.
                 return None # Or raise a specific exception FastAPI ignores on disconnect
            except Exception as e:
                 # Catch any other unexpected error during setup/dispatch
                 logging.exception("Error processing completion request.")
                 # Ensure stop is attempted if something failed mid-way
                 if page and config:
                     await stop_generation(page, config)
                 raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        except PlaywrightError as e:
            logging.error(f"Playwright error during interaction: {e}")
            raise HTTPException(status_code=500, detail=f"Browser interaction error: {e}")
        except Exception as e:
            logging.exception("Unexpected error in completion endpoint.")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
        finally:
             # Ensure task is cleared if an exception happened *before* it was awaited/returned
             if app_state.get("current_task") and not app_state["current_task"].done():
                  app_state["current_task"].cancel() # Attempt cancellation
             app_state["current_task"] = None
             logging.debug("Completion request handler finished.")


# --- Signal Handling for Graceful Shutdown ---
def handle_signal(sig, frame):
    logging.warning(f"Received signal {sig}. Initiating graceful shutdown.")
    # This doesn't directly stop Uvicorn, but lifespan shutdown will be triggered
    # You might need to send SIGINT/SIGTERM to the Uvicorn process itself
    # For simplicity, we rely on standard Ctrl+C / kill signals triggering lifespan 'shutdown'
    sys.exit(0) # Signal Uvicorn to shut down if running directly

# --- Main Execution ---
if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown (optional but good practice)
    # signal.signal(signal.SIGINT, handle_signal)
    # signal.signal(signal.SIGTERM, handle_signal)

    # Make sure Playwright browsers are installed
    # Run `playwright install` in your terminal if you haven't already
    logging.info("Checking Playwright installation...")
    try:
        # This is a blocking call, run before async loop
        import subprocess
        process = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True, text=True, check=False)
        if process.returncode != 0:
             # Don't necessarily fail if install fails (maybe already installed system-wide)
             logging.warning(f"Playwright install command finished with code {process.returncode}.")
             # print(process.stdout)
             # print(process.stderr)
        else:
             logging.info("Playwright browser check/install completed.")
    except Exception as e:
        logging.warning(f"Could not automatically run 'playwright install': {e}. Please ensure browsers are installed manually.")


    print("Starting application...")
    # Uvicorn will manage the asyncio loop and run the lifespan manager
    uvicorn.run(app, host="127.0.0.1", port=9223)

    # Code here might not be reached if Uvicorn takes over the loop completely
    print("Application stopped.")