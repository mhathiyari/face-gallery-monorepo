const state = {
  collections: [],
  activeCollection: null,
  people: [],
  activePerson: null,
  ownerMode: false,
  currentJobId: null,
  cropMode: true,
  shares: { collection_link: null, people: {} },
};

const el = (id) => document.getElementById(id);

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadCollections() {
  state.collections = await fetchJSON("/api/collections");
  state.activeCollection = await fetchJSON("/api/collections/active");
  renderCollections();
}

async function loadSettings() {
  const settings = await fetchJSON("/api/settings");
  if (settings.last_input_folder) {
    el("inputFolder").value = settings.last_input_folder;
  }
  if (settings.last_output_folder) {
    el("outputFolder").value = settings.last_output_folder;
  }
}

async function loadShares() {
  state.shares = await fetchJSON("/api/shares");
  renderShares();
  renderPeople();
}

function renderCollections() {
  const list = el("collectionsList");
  list.innerHTML = "";
  state.collections.forEach((collection) => {
    const item = document.createElement("div");
    item.className = "collection-item";
    item.addEventListener("click", () => setActiveCollection(collection.id));

    const info = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = collection.name;
    const meta = document.createElement("span");
    meta.textContent = `${collection.person_count} people | ${collection.image_count} images`;
    info.appendChild(title);
    info.appendChild(meta);

    item.appendChild(info);
    if (state.activeCollection && state.activeCollection.id === collection.id) {
      const removeButton = document.createElement("button");
      removeButton.className = "collection-remove";
      removeButton.textContent = "Remove";
      removeButton.addEventListener("click", (event) => {
        event.stopPropagation();
        removeCollection(collection.id, collection.name).catch(showError);
      });
      item.appendChild(removeButton);
      item.classList.add("collection-active");
    }
    list.appendChild(item);
  });
}

async function setActiveCollection(collectionId) {
  await fetchJSON("/api/collections/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: collectionId }),
  });
  await loadCollections();
  await loadPeople();
}

async function addCollection() {
  const path = el("collectionPath").value.trim();
  if (!path) return;
  await fetchJSON("/api/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  el("collectionPath").value = "";
  await loadCollections();
  await loadPeople();
}

async function removeCollection(collectionId, name) {
  const confirmed = window.confirm(`Remove collection \"${name}\" from the list?`);
  if (!confirmed) return;
  await fetchJSON(`/api/collections/${collectionId}`, { method: "DELETE" });
  await loadCollections();
  await loadPeople();
}

async function loadPeople() {
  if (!state.activeCollection) {
    state.people = [];
    renderPeople();
    return;
  }
  const sort = el("sortMode").value;
  state.people = await fetchJSON(`/api/people?sort=${sort}`);
  renderPeople();
}

function renderPeople() {
  const grid = el("peopleGrid");
  grid.innerHTML = "";
  state.people.forEach((person) => {
    const card = document.createElement("div");
    card.className = "card";
    card.addEventListener("click", () => loadPersonDetail(person.id));

    const img = document.createElement("img");
    if (person.representative && state.activeCollection) {
      if (person.id === "unmatched") {
        img.src = `/media/${state.activeCollection.id}/${person.id}/${person.representative}`;
      } else {
        img.src = `/face-rep/${state.activeCollection.id}/${person.id}?v=${Date.now()}`;
      }
    } else {
      img.alt = "No representative";
    }

    const title = document.createElement("h3");
    title.textContent = person.label;

    const meta = document.createElement("p");
    meta.textContent = `${person.count} photos`;

    const source = document.createElement("span");
    source.className = "rep-source";
    source.textContent = person.representative_source || "unknown";

    card.appendChild(img);
    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(source);
    const link = state.shares.people[person.id];
    if (link) {
      const linkButton = document.createElement("button");
      linkButton.textContent = "Copy link";
      linkButton.addEventListener("click", (event) => {
        event.stopPropagation();
        copyText(link);
      });
      card.appendChild(linkButton);
    }
    grid.appendChild(card);
  });
}

async function loadPersonDetail(personId) {
  if (!state.activeCollection) return;
  state.activePerson = personId;
  const payload = await fetchJSON(`/api/person/${personId}?page=1&page_size=80`);
  const meta = el("personMeta");
  meta.textContent = `${payload.total} images in ${personId}`;
  const person = state.people.find((item) => item.id === personId);
  el("personLabel").value = person ? person.label : personId;

  const imagesGrid = el("personImages");
  imagesGrid.innerHTML = "";
  payload.images.forEach((filename) => {
    const img = document.createElement("img");
    const useCrop = state.cropMode && personId !== "unmatched";
    const base = useCrop ? "face" : "media";
    img.src = `/${base}/${state.activeCollection.id}/${personId}/${filename}`;
    img.alt = filename;
    if (state.ownerMode) {
      img.addEventListener("click", () => setRepresentative(personId, filename));
    }
    imagesGrid.appendChild(img);
  });
}

async function saveLabel() {
  if (!state.ownerMode || !state.activePerson) return;
  const label = el("personLabel").value.trim();
  if (!label) return;
  await fetchJSON(`/api/person/${state.activePerson}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  await loadPeople();
}

async function setRepresentative(personId, filename) {
  if (!state.ownerMode) return;
  await fetchJSON(`/api/person/${personId}/representative`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  await loadPeople();
}

async function startJob() {
  if (!state.ownerMode) return;
  const inputFolder = el("inputFolder").value.trim();
  const outputFolder = el("outputFolder").value.trim();
  if (!inputFolder || !outputFolder) return;

  const response = await fetchJSON("/api/jobs/sort", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input_folder: inputFolder, output_folder: outputFolder }),
  });
  state.currentJobId = response.job_id;
  el("jobState").textContent = `Job ${state.currentJobId} running...`;
  pollJob();
}

async function pollJob() {
  if (!state.currentJobId) return;
  const payload = await fetchJSON(`/api/jobs/${state.currentJobId}`);
  el("jobState").textContent = `Job status: ${payload.status}`;
  el("jobLogs").textContent = payload.logs.join("\n");
  if (payload.status === "running") {
    setTimeout(pollJob, 2000);
  }
}

async function runSearch() {
  if (!state.activeCollection) return;
  const fileInput = el("searchImage");
  if (!fileInput.files.length) return;

  const form = new FormData();
  form.append("image", fileInput.files[0]);

  const response = await fetch("/api/search", { method: "POST", body: form });
  const payload = await response.json();
  const result = el("searchResult");

  if (!response.ok) {
    result.textContent = payload.error || "Search failed";
    return;
  }

  if (!payload.match) {
    result.textContent = "No match found.";
    return;
  }

  const match = payload.match;
  const lines = [
    `Match: ${match.person_folder}`,
    `Confidence: ${(match.confidence * 100).toFixed(1)}%`,
    `Matches: ${match.match_count}`,
  ];
  if (payload.top_matches && payload.top_matches.length) {
    lines.push("\nTop matches:");
    payload.top_matches.forEach((item, index) => {
      lines.push(`${index + 1}. ${item.folder} (${(item.avg_similarity * 100).toFixed(1)}%)`);
    });
  }
  result.textContent = lines.join("\n");
}

function setOwnerMode(enabled) {
  state.ownerMode = enabled;
  el("saveLabel").disabled = !enabled;
  el("startJob").disabled = !enabled;
  el("personLabel").disabled = !enabled;
  if (!state.currentJobId) {
    const note = enabled ? "Owner mode enabled" : "Owner mode disabled";
    el("jobState").textContent = note;
  }
  if (state.activePerson) {
    loadPersonDetail(state.activePerson).catch(showError);
  }
}

function wireEvents() {
  el("addCollection").addEventListener("click", () => addCollection().catch(showError));
  el("refreshPeople").addEventListener("click", () => loadPeople().catch(showError));
  el("sortMode").addEventListener("change", () => loadPeople().catch(showError));
  el("saveLabel").addEventListener("click", () => saveLabel().catch(showError));
  el("startJob").addEventListener("click", () => startJob().catch(showError));
  el("runSearch").addEventListener("click", () => runSearch().catch(showError));
  el("ownerMode").addEventListener("change", (event) => setOwnerMode(event.target.checked));
  el("uploadGallery").addEventListener("click", () => uploadDrive("gallery").catch(showError));
  el("uploadPeople").addEventListener("click", () => uploadDrive("people").catch(showError));
  el("cropMode").addEventListener("change", (event) => {
    state.cropMode = event.target.checked;
    if (state.activePerson) {
      loadPersonDetail(state.activePerson).catch(showError);
    }
  });
}

async function uploadDrive(mode) {
  el("shareStatus").textContent = "Uploading to Google Drive...";
  const response = await fetchJSON("/api/drive/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  el("shareStatus").textContent = `Upload complete. ${response.people_count} people linked.`;
  await loadShares();
}

function renderShares() {
  const container = el("shareLinks");
  container.innerHTML = "";
  if (!state.shares.collection_link) {
    el("shareStatus").textContent = "No uploads yet.";
    return;
  }
  el("shareStatus").textContent = "Drive links ready.";
  const row = document.createElement("div");
  row.className = "share-link";
  const label = document.createElement("span");
  label.textContent = "Collection link";
  const button = document.createElement("button");
  button.textContent = "Copy";
  button.addEventListener("click", () => copyText(state.shares.collection_link));
  row.appendChild(label);
  row.appendChild(button);
  container.appendChild(row);
}

function copyText(text) {
  navigator.clipboard.writeText(text).catch(() => {});
}

function showError(error) {
  const result = el("searchResult");
  result.textContent = error.message;
}

async function init() {
  wireEvents();
  setOwnerMode(false);
  await loadSettings();
  await loadCollections();
  await loadShares();
  await loadPeople();
}

init().catch(showError);
