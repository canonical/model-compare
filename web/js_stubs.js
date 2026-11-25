// js_stubs.js — minimal browser shims for evaluating the site's app.js
// under node (web/test_site_js.py prepends this file to web/site/app.js).
//
// Only what the script touches at evaluation time: element lookups,
// tab/expander wiring, and the four fetch() calls (rejected, so the
// page ends in its error path and pure helpers stay callable).
//
// Limitation: these shims support evaluating the script's pure helpers
// (tooltip sentence construction, formatters, guards), the built-element
// shape of endpointRow()/baselineNote() and of render() against a stub
// tbody — appendChild records children so a driver can walk the DOM the
// script built, including the title a render loop assigned to a cell —
// named mutations like renderFooter() (getElementById hands out the same
// cached stub per id, so a driver can read back what the script wrote
// through els or document.getElementById), and the event wiring:
// addEventListener records handlers per type and dispatchEvent fires
// them, so a driver can click the view/consent buttons. The live fetch
// flows are not exercised: fetch() rejects so render() never runs from
// the fetch path, and querySelectorAll returns [].

function stubElement() {
  var el = {
    hidden: false,
    disabled: false,
    title: "",
    className: "",
    colSpan: 0,
    tabIndex: 0,
    children: [],
    dataset: {},
    style: {},
    // Listeners recorded per type so a driver can fire events and a
    // test can prove which handler is bound to which element (the
    // consent wiring tests depend on this).
    listeners: {},
    attrs: {},
    addEventListener: function (type, fn) {
      (this.listeners[type] = this.listeners[type] || []).push(fn);
    },
    dispatchEvent: function (type, extra) {
      var ev = { type: type, preventDefault: function () {} };
      if (extra) {
        for (var k in extra) ev[k] = extra[k];
      }
      (this.listeners[type] || []).forEach(function (fn) {
        fn.call(this, ev);
      }, this);
    },
    setAttribute: function (name, value) {
      this.attrs[name] = String(value);
    },
    getAttribute: function (name) {
      return Object.prototype.hasOwnProperty.call(this.attrs, name)
        ? this.attrs[name]
        : null;
    },
    focus: function () {
      document.activeElement = this;
    },
    querySelector: function () {
      return null;
    },
    querySelectorAll: function () {
      return [];
    },
    classList: {
      _set: {},
      toggle: function (name, force) {
        var on = force === undefined ? !this._set[name] : !!force;
        if (on) this._set[name] = true;
        else delete this._set[name];
      },
      contains: function (name) {
        return !!this._set[name];
      },
      add: function (name) {
        this._set[name] = true;
      },
      remove: function (name) {
        delete this._set[name];
      },
    },
    setAttribute: function (name, value) {
      this.attrs[name] = String(value);
    },
    getAttribute: function (name) {
      return Object.prototype.hasOwnProperty.call(this.attrs, name)
        ? this.attrs[name]
        : null;
    },
    appendChild: function (child) {
      this.children.push(child);
      return child;
    },
    remove: function () {},
    select: function () {},
    open: false,
    showModal: function () {
      if (this.open) throw new Error("InvalidStateError: dialog already open");
      this.open = true;
    },
    close: function () {
      this.open = false;
    },
  };
  // Real DOM: assigning textContent replaces all child nodes. The old stub
  // left them in place, so render()'s "clear the table" could never be
  // observed and stale rows survived every re-render.
  var text = "";
  Object.defineProperty(el, "textContent", {
    get: function () {
      return text;
    },
    set: function (value) {
      text = String(value);
      this.children = [];
    },
  });
  return el;
}

// One stub per element id: the script captures its elements once (the
// els object) and later mutations must stay observable through both els
// and re-lookups via document.getElementById.
var stubsById = {};
var window = { location: { href: "http://127.0.0.1:8734/index.html" } };
var document = {
  // cookie is a real jar below, so two preference cookies can coexist.
  activeElement: null,
  getElementById: function (id) {
    if (!Object.prototype.hasOwnProperty.call(stubsById, id)) {
      stubsById[id] = stubElement();
    }
    return stubsById[id];
  },
  createElement: function () {
    return stubElement();
  },
  createTextNode: function (text) {
    return { textContent: String(text) };
  },
  querySelectorAll: function () {
    return [];
  },
  querySelector: function () {
    return null;
  },
  body: stubElement(),
};
// A minimal cookie jar: the site stores one layout cookie and one retention
// cookie, so a single overwritten string hid cross-cookie bugs.
var cookieJar = {};
// Raw writes are recorded for tests that need to assert the attributes
// (real document.cookie only exposes name=value pairs). An empty write
// clears the jar: that is a test convenience standing in for a browser
// cleared between page loads, not real cookie syntax.
var cookieWrites = [];
document.__cookieWrites = cookieWrites;
Object.defineProperty(document, "cookie", {
  get: function () {
    return Object.keys(cookieJar)
      .map(function (name) {
        return name + "=" + cookieJar[name];
      })
      .join("; ");
  },
  set: function (raw) {
    raw = String(raw);
    cookieWrites.push(raw);
    if (raw === "") {
      cookieJar = {};
      return;
    }
    var first = raw.split(";")[0];
    var eq = first.indexOf("=");
    if (eq < 0) return;
    var name = first.slice(0, eq).trim();
    if (!name) return;
    if (/max-age\s*=\s*0\b/i.test(String(raw))) {
      delete cookieJar[name];
      return;
    }
    cookieJar[name] = first.slice(eq + 1).trim();
  },
});
var navigator = {};
var fetch = function () {
  return Promise.reject(new Error("stub: fetch disabled in JS tests"));
};
