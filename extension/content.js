(function () {
  console.log("Facebook Post Interceptor Injected!");

  let posts = {};
  window.getPosts = () => posts;

  function extractPosts(obj) {
    if (!obj) return;
    if (typeof obj !== "object") {
      return;
    }

    if (
      obj.comet_sections &&
      obj.comet_sections.content &&
      obj.comet_sections.content.story
    ) {
      const story = obj.comet_sections.content.story;
      try {
        const id = story["post_id"];

        if (id) {
          const text = story.message?.text || "";
          const actor = story.actors[0];
          let post = {
            id: id,
            // data: story,
            author: actor,
            text: text,
            url: story.wwwURL,
            done: false,
          };
          posts[post.id] = post;
          console.log({ post });
        }
      } catch {
        console.warn({ story });
      }
    }

    // Recursively traverse the JSON tree to find nested posts (e.g., in feeds or comments)
    for (const key in obj) {
      if (
        Object.prototype.hasOwnProperty.call(obj, key) &&
        typeof obj[key] === "object"
      ) {
        extractPosts(obj[key]);
      }
    }
  }

  // Facebook GraphQL responses are often multi-part/line-delimited JSON
  function processGraphQLResponse(responseText) {
    if (!responseText) return;
    const lines = responseText.split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      const data = JSON.parse(line);
      extractPosts(data);
    }
  }

  const originalXHR = {
    open: XMLHttpRequest.prototype.open,
    send: XMLHttpRequest.prototype.send,
  };

  window.XMLHttpRequest.prototype.open = function (method, url) {
    console.log("opened request " + url);
    this._url = url;
    return originalXHR.open.apply(this, arguments);
  };

  window.XMLHttpRequest.prototype.send = function () {
    this.addEventListener("load", function () {
      if (this._url.includes("/graphql/")) {
        console.log("intercepting.... " + this._url);
        processGraphQLResponse(this.responseText);
      }
    });
    this.addEventListener("error", () => {
      console.log("XHR error", {
        url: this._url,
        status: this.status,
        statusText: this.statusText,
        responseURL: this.responseURL,
      });
    });

    return originalXHR.send.apply(this, arguments);
  };

  // TODO: add fetch if needed, clean up the code.
  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const request = args[0];
    console.log({ request });
    let urlStr = "";

    if (typeof request === "string") {
      urlStr = request;
    } else if (request instanceof Request) {
      urlStr = request.url;
    } else if (request && request.url) {
      urlStr = request.url;
    } else if (request) {
      urlStr = request.toString();
    }

    const response = await originalFetch.apply(this, args);

    if (urlStr && urlStr.includes("/graphql/")) {
      try {
        const clonedResponse = response.clone();
        clonedResponse
          .text()
          .then((text) => {
            processGraphQLResponse(text);
          })
          .catch((e) => {
            /* Ignore clone read errors */
          });
      } catch (e) {
        console.error("Error cloning fetch response:", e);
      }
    }
    return response;
  };

  const fetches = new Map();
  let fetchId = 0;

  function fgfetch(...args) {
    return new Promise((resolve, reject) => {
      const id = ++fetchId;
      fetches.set(id, { resolve, reject });
      window.dispatchEvent(
        new CustomEvent("fgfetch", { detail: { id, args } }),
      );
    });
  }
  window.addEventListener("fgfetch:done", ({ detail }) => {
    const fetch = fetches.get(detail.id);
    if (!fetch) return;
    fetches.delete(detail.id);
    detail.error
      ? fetch.reject(new Error(detail.error))
      : fetch.resolve(new Response(detail.body, detail.response));
  });

  function findFeedUnitFiber(rootFiber, pos) {
    if (!rootFiber) return null;

    const stack = [rootFiber];

    while (stack.length > 0) {
      const currentFiber = stack.pop();
      const props = currentFiber.memoizedProps;

      if (
        props &&
        typeof props === "object" &&
        "feedUnit" in props &&
        "position" in props &&
        props.position == pos
        // props.position
      ) {
        // console.log({ feedUnit: props.feedUnit, props: props });
        return props.feedUnit["post_id"];
      }

      if (currentFiber.sibling) {
        stack.push(currentFiber.sibling);
      }

      if (currentFiber.child) {
        stack.push(currentFiber.child);
      }
    }

    return null;
  }
  async function handlePost(post) {
    // console.log(post.ariaPosInSet);

    function getId(post) {
      const cachedId = post.getAttribute("post-id");
      if (cachedId) return cachedId;

      const rootFiber = getRootFiber();
      const id = findFeedUnitFiber(rootFiber, parseInt(post.ariaPosInSet) - 1);
      if (id) {
        post.setAttribute("post-id", id);
      }
      return id;
    }
    const id = getId(post);
    if (!id) {
      post.classList.add("factguard-post-unknown");
      return;
    }
    if (!posts[id]) {
      // console.warn(`could not find post ${id} in the intercepted posts`);
      post.classList.remove("factguard-post-unknown");
      post.classList.add("factguard-post-unintercepted");
      return;
    }
    post.classList.remove("factguard-post-unknown");
    post.classList.remove("factguard-post-unintercepted");
    post.classList.add("factguard-post");

    console.log(`handled post ${id}`);
    // console.log({
    //   post,
    //   maybeId: id,
    //   maybePostObject: posts[id],
    // });
    const postText = posts[id].text;

    const reportMisinfoButton = document.createElement("button");
    reportMisinfoButton.classList.add("factguard-post-misinformation-button");
    reportMisinfoButton.innerText = "Report misinformation";
    reportMisinfoButton.onclick = async (event) => {
      const btn = event.target;
      console.log("reporting ", id);
      try {
        const reportResponse = await fgfetch("http://localhost:5000/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: postText,
          }),
        });
        const result = await reportResponse.json();
        if (result.success) {
          btn.innerText = "Reported";
          btn.disabled = true;
          post.classList.add("factguard-post-misinformation");
        }
      } catch (error) {
        console.error("Failed to report post:", error);
      }
    };
    post.append(reportMisinfoButton);
    const response = await fgfetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: postText,
      }),
    });
    const score = (await response.json()).score;
    if (score > 0.95) {
      post.classList.add("factguard-post-misinformation");
    }
    // TODO: add report misinformation button
    console.log({
      id: id,
      text: postText,
      score,
    });
    delete posts[id];
  }

  function getRootFiber() {
    const rootElement = document.querySelector('[id^="mount_0_0_"]');
    for (const key in rootElement) {
      if (key.startsWith("__reactContainer$")) {
        return rootElement[key];
      }
    }
  }

  function callback(mutationList, _observer) {
    for (const mutation of mutationList) {
      if (mutation.type === "childList") {
        const posts = document.querySelectorAll(
          'div[aria-posinset]:not(.factguard-post):has(div[aria-label="Like"])',
        );
        for (const postNode of posts) {
          handlePost(postNode);
        }
      }
    }
  }
  const observer = new MutationObserver(callback);

  observer.observe(document, { childList: true, subtree: true });
})();
