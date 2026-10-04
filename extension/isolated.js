window.addEventListener("fgfetch", async ({ detail }) => {
  function done(payload) {
    window.dispatchEvent(
      new CustomEvent("fgfetch:done", {
        detail: { id: detail.id, ...payload },
      }),
    );
  }

  try {
    const response = await fetch(...detail.args);
    done({
      body: await response.text(),
      response: {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers),
      },
    });
  } catch (error) {
    done({ error: error.message });
  }
});
