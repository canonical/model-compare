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
  return {
    textContent: "",
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
}

// One stub per element id: the script captures its elements once (the
// els object) and later mutations must stay observable through both els
// and re-lookups via document.getElementById.
var stubsById = {};
var window = { location: { href: "http://127.0.0.1:8734/index.html" } };
var document = {
  // cookie: the consent-first view preference is read at init() time;
  // empty by default, tests may assign a header-shaped string.
  cookie: "",
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
  querySelectorAll: function () {
    return [];
  },
  querySelector: function () {
    return null;
  },
  body: stubElement(),
};
var navigator = {};
var fetch = function () {
  return Promise.reject(new Error("stub: fetch disabled in JS tests"));
};
