/**
 * Compatibility barrel for toml actions + training launch helpers.
 * Domain truth lives in feature modules; importing this file still configures bridges.
 */
export * from '../../toml-manager/actions.js?v=module-bootstrap-20260711-ir2';
export * from '../../training-launch/index.js?v=module-bootstrap-20260711-ir2';
